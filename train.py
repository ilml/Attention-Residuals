"""
Training script for Attention Residuals scaling law experiments.
Supports multi-node DDP with bf16 mixed precision.

Usage (single node, 4 GPUs):
  torchrun --nproc_per_node=4 train.py --config 124M --variant baseline

Usage (multi-node via Slurm, see submit_large.sh):
  srun torchrun ... train.py --config 401M --variant block_attnres --large
"""

import os
import math
import time
import json
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

import wandb
from model import Transformer
from data import get_dataloaders, VOCAB_SIZE, SEQ_LEN
from configs import SCALING_CONFIGS, SCALING_CONFIGS_SMALL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Model size config name (e.g., 124M, 172M, ...)")
    p.add_argument("--variant", type=str, required=True,
                   choices=["baseline", "full_attnres", "block_attnres"])
    p.add_argument("--large", action="store_true",
                   help="Use large-scale configs (default: small/sample configs)")
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--data_dir", type=str, default=None,
                   help="Override data directory")
    p.add_argument("--wandb_project", type=str, default="attenres")
    p.add_argument("--wandb_entity", type=str, default="tom-vita")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--micro_batch", type=int, default=None)
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_interval", type=int, default=500,
                   help="Save checkpoint every N steps (0=disabled)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def get_cosine_schedule(step, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1):
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, val_loader, device, ctx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with ctx:
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    model.train()
    stats = torch.tensor([total_loss, total_tokens], dtype=torch.float64, device=device)
    dist.all_reduce(stats)
    return (stats[0] / stats[1]).item()


def save_checkpoint(model, optimizer, step, tokens_seen, args, cfg, path):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "config": args.config,
        "variant": args.variant,
        "cfg": cfg,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt["tokens_seen"]


def auto_micro_batch(variant: str, n_layer: int, world_size: int, global_batch: int) -> int:
    """Pick micro_batch based on variant, model size, and GPU count.
    With many GPUs (e.g. 256), per_gpu can be <1, so we always use at least 1
    and rely on grad_accum to reach the target batch size."""
    per_gpu = max(1, global_batch // world_size)
    if variant == "full_attnres":
        if n_layer >= 17:
            mb = 1
        elif n_layer >= 14:
            mb = 2
        else:
            mb = 4
    elif variant == "block_attnres":
        if n_layer >= 17:
            mb = 2
        elif n_layer >= 14:
            mb = 4
        else:
            mb = 6
    else:
        mb = min(16, per_gpu)
    # With many GPUs, per_gpu may be 0; ensure at least 1
    return max(1, min(mb, max(1, per_gpu)))


def train():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0

    # Pick config set
    configs = SCALING_CONFIGS if args.large else SCALING_CONFIGS_SMALL
    if args.config not in configs:
        if is_master:
            print(f"ERROR: config '{args.config}' not found. Available: {list(configs.keys())}")
        dist.destroy_process_group()
        return

    # Skip if results already exist
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = os.path.join(args.save_dir, f"{args.config}_{args.variant}_results.pt")
    if os.path.exists(result_file):
        if is_master:
            existing = torch.load(result_file, map_location="cpu", weights_only=False)
            print(f"SKIPPING {args.config}/{args.variant}: results exist "
                  f"(val_loss={existing['val_loss']:.4f})")
        dist.destroy_process_group()
        return

    # Seed
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    cfg = configs[args.config]
    max_steps = args.max_steps or cfg["max_steps"]
    warmup_steps = max(1, int(max_steps * 0.1))
    global_batch = cfg["batch_size"]

    # Micro batch
    if args.micro_batch is not None:
        micro_batch = args.micro_batch
    else:
        micro_batch = auto_micro_batch(args.variant, cfg["n_layer"], world_size, global_batch)
    grad_accum_steps = max(1, global_batch // (micro_batch * world_size))
    effective_batch = micro_batch * world_size * grad_accum_steps

    if is_master:
        print(f"{'='*60}")
        print(f"Config: {args.config} | Variant: {args.variant} | {'LARGE' if args.large else 'SMALL'}")
        print(f"World: {world_size} GPUs | Micro batch: {micro_batch}/GPU | "
              f"Grad accum: {grad_accum_steps} | Effective batch: {effective_batch}")
        print(f"Steps: {max_steps} | Warmup: {warmup_steps} | "
              f"Tokens: {effective_batch * 8192 * max_steps / 1e9:.1f}B")
        print(f"{'='*60}")

    # Model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_ff=cfg["d_ff"],
        max_seq_len=SEQ_LEN,
        residual_mode=args.variant,
        num_blocks=args.num_blocks,
    ).to(device)

    if is_master:
        print(f"Parameters: {model.count_parameters():,}")

    if args.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)

    # Resume from checkpoint (auto-detect latest if not specified)
    start_step = 0
    tokens_seen = 0
    resume_path = args.resume
    if not resume_path:
        # Auto-find latest checkpoint for this config/variant (sort by step number)
        import glob as _glob, re as _re
        pattern = os.path.join(args.save_dir, f"{args.config}_{args.variant}_step*.pt")
        ckpts = _glob.glob(pattern)
        if ckpts:
            # Sort numerically by step number, not alphabetically
            ckpts.sort(key=lambda p: int(_re.search(r"step(\d+)", p).group(1)))
            resume_path = ckpts[-1]  # highest step number
    if resume_path and os.path.exists(resume_path):
        start_step, tokens_seen = load_checkpoint(resume_path, model, optimizer, device)
        if is_master:
            print(f"RESUMED from {resume_path} at step {start_step}")

    ctx = autocast(device_type="cuda", dtype=torch.bfloat16)

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=micro_batch, seq_len=SEQ_LEN,
        rank=rank, world_size=world_size,
        num_workers=4, data_dir=args.data_dir)

    # wandb — enabled by default, set WANDB_MODE=offline to disable cloud sync
    use_wandb = False
    if is_master:
        run_name = f"{args.config}_{args.variant}"
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                group=args.config,       # group runs by model size
                tags=[args.variant, args.config,
                      "large" if args.large else "small"],
                config={
                    "model_config": args.config,
                    "variant": args.variant,
                    "d_model": cfg["d_model"],
                    "n_layer": cfg["n_layer"],
                    "n_head": cfg["n_head"],
                    "d_ff": cfg["d_ff"],
                    "lr": cfg["lr"],
                    "global_batch_size": effective_batch,
                    "micro_batch": micro_batch,
                    "grad_accum_steps": grad_accum_steps,
                    "max_steps": max_steps,
                    "warmup_steps": warmup_steps,
                    "world_size": world_size,
                    "num_params": model.module.count_parameters(),
                    "large": args.large,
                    "seq_len": SEQ_LEN,
                    "num_blocks": args.num_blocks,
                },
            )
            # Define metric sections for grouped panels
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="train/step")
            wandb.define_metric("perf/*", step_metric="train/step")
            use_wandb = True
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}), continuing without wandb")

    # Training loop
    model.train()
    optimizer.zero_grad()
    step = start_step
    train_iter = iter(train_loader)
    running_loss = 0.0
    t_start = time.time()

    while step < max_steps:
        lr = get_cosine_schedule(step, warmup_steps, max_steps, cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_loader.sampler.set_epoch(step)
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            inputs, targets = inputs.to(device), targets.to(device)
            sync_ctx = model.no_sync if micro_step < grad_accum_steps - 1 else nullcontext
            with sync_ctx():
                with ctx:
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1))
                    scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()
            accum_loss += loss.item()
            tokens_seen += inputs.numel() * world_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_step_loss = accum_loss / grad_accum_steps
        running_loss += avg_step_loss
        step += 1

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            tps = tokens_seen / elapsed
            avg_loss = running_loss / args.log_interval
            raw_model = model.module if hasattr(model, "module") else model
            pflops_days = 6 * raw_model.flops_per_token() * tokens_seen / (1e15 * 86400)

            if is_master:
                if use_wandb:
                    wandb.log({
                        "train/step": step,
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/tokens_seen": tokens_seen,
                        "perf/tokens_per_sec": tps,
                        "perf/pflops_days": pflops_days,
                    }, step=step)
                print(f"Step {step}/{max_steps} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"tok/s {tps:.0f} | PFLOP/s-days {pflops_days:.4f}")
            running_loss = 0.0

        # Validation
        if step % args.val_interval == 0 or step == max_steps:
            val_loss = validate(model, val_loader, device, ctx)
            raw_model = model.module if hasattr(model, "module") else model
            pflops_days = 6 * raw_model.flops_per_token() * tokens_seen / (1e15 * 86400)
            if is_master:
                if use_wandb:
                    wandb.log({
                        "train/step": step,
                        "val/loss": val_loss,
                        "val/pflops_days": pflops_days,
                    }, step=step)
                print(f"  >> Val loss: {val_loss:.4f} at {pflops_days:.4f} PFLOP/s-days")

        # Checkpoint
        if args.save_interval > 0 and step % args.save_interval == 0 and is_master:
            ckpt_path = os.path.join(args.save_dir, f"{args.config}_{args.variant}_step{step}.pt")
            save_checkpoint(model, optimizer, step, tokens_seen, args, cfg, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final save
    if is_master:
        raw_model = model.module if hasattr(model, "module") else model
        pflops_days = 6 * raw_model.flops_per_token() * tokens_seen / (1e15 * 86400)

        results = {
            "config": args.config,
            "variant": args.variant,
            "val_loss": val_loss,
            "pflops_days": pflops_days,
            "tokens_seen": tokens_seen,
            "num_params": raw_model.count_parameters(),
            "world_size": world_size,
            "large": args.large,
        }
        torch.save(results, result_file)
        print(f"\nFinal: {args.config}/{args.variant} val_loss={val_loss:.4f} "
              f"pflops_days={pflops_days:.4f}")

        if use_wandb:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
