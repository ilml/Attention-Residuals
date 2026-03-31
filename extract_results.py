"""
Extract results from checkpoints by running validation on a single GPU.
For when multi-node training completes but crashes before saving result files.

Usage:
  python extract_results.py --save_dir checkpoints_paper --data_dir /path/to/data
"""

import os
import re
import glob
import json
import argparse
import torch
from torch.amp import autocast

from model import Transformer
from data import get_dataloaders, VOCAB_SIZE, SEQ_LEN
from configs import SCALING_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="checkpoints_paper")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()

    configs = SCALING_CONFIGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()

    # Find all checkpoint files and group by config/variant
    ckpt_pattern = os.path.join(args.save_dir, "*_step*.pt")
    all_ckpts = glob.glob(ckpt_pattern)

    experiments = {}
    for ckpt in all_ckpts:
        basename = os.path.basename(ckpt)
        match = re.match(r"(.+?)_(.+?)_step(\d+)\.pt", basename)
        if match:
            config, variant, step = match.group(1), match.group(2), int(match.group(3))
            key = (config, variant)
            if key not in experiments or step > experiments[key][1]:
                experiments[key] = (ckpt, step)

    print(f"Found {len(experiments)} experiments with checkpoints:")
    for (cfg, var), (path, step) in sorted(experiments.items()):
        result_file = os.path.join(args.save_dir, f"{cfg}_{var}_results.pt")
        status = "HAS RESULT" if os.path.exists(result_file) else "NEEDS EVAL"
        total = configs[cfg]["max_steps"] if cfg in configs else "?"
        print(f"  {cfg:>5} {var:<15} step {step:>6}/{total} -> {status}")

    # Load data once
    print("\nLoading validation data...")
    _, val_loader = get_dataloaders(
        batch_size=4, seq_len=SEQ_LEN, rank=0, world_size=1,
        num_workers=2, data_dir=args.data_dir)

    # Process each experiment
    for (cfg, var), (ckpt_path, step) in sorted(experiments.items()):
        result_file = os.path.join(args.save_dir, f"{cfg}_{var}_results.pt")
        if os.path.exists(result_file):
            existing = torch.load(result_file, map_location="cpu", weights_only=False)
            print(f"\n{cfg}/{var}: already has result (val={existing['val_loss']:.4f})")
            continue

        if cfg not in configs:
            print(f"\n{cfg}/{var}: config not found, skipping")
            continue

        c = configs[cfg]
        total_steps = c["max_steps"]
        progress = step / total_steps * 100

        print(f"\n{cfg}/{var}: loading checkpoint step {step}/{total_steps} ({progress:.0f}%)...")

        model = Transformer(
            vocab_size=VOCAB_SIZE, d_model=c["d_model"], n_layer=c["n_layer"],
            n_head=c["n_head"], d_ff=c["d_ff"], max_seq_len=SEQ_LEN,
            residual_mode=var, num_blocks=8,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        tokens_seen = ckpt.get("tokens_seen", step * c["batch_size"] * SEQ_LEN)

        # Run validation
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with ctx:
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()

        val_loss = total_loss / total_tokens
        pflops_days = 6 * model.flops_per_token() * tokens_seen / (1e15 * 86400)

        results = {
            "config": cfg, "variant": var, "val_loss": val_loss,
            "pflops_days": pflops_days, "tokens_seen": tokens_seen,
            "num_params": model.count_parameters(),
            "step": step, "total_steps": total_steps,
        }
        torch.save(results, result_file)
        print(f"  val_loss={val_loss:.4f} at step {step}/{total_steps} -> {result_file}")
        del model, ckpt
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
