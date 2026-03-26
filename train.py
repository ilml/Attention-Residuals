"""
Training script for Attention Residuals scaling law experiments.
Uses PyTorch DDP on 4 GPUs with bf16 mixed precision.

Usage:
  torchrun --nproc_per_node=4 train.py \
      --config 194M --variant baseline --wandb_project attn-residuals
"""

import os
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

import wandb
from model import Transformer
from data import get_dataloaders, VOCAB_SIZE, SEQ_LEN
from configs import SCALING_CONFIGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, choices=list(SCALING_CONFIGS.keys()),
                   help="Model size config name")
    p.add_argument("--variant", type=str, required=True,
                   choices=["baseline", "full_attnres", "block_attnres"],
                   help="Residual connection variant")
    p.add_argument("--num_blocks", type=int, default=8,
                   help="Number of blocks for Block AttnRes")
    p.add_argument("--wandb_project", type=str, default="attn-residuals")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override max training steps (default: from config)")
    p.add_argument("--micro_batch", type=int, default=None,
                   help="Per-GPU micro batch size (default: auto)")
    p.add_argument("--val_interval", type=int, default=100,
                   help="Validate every N optimizer steps")
    p.add_argument("--log_interval", type=int, default=10,
                   help="Log every N optimizer steps")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    return p.parse_args()


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def get_cosine_schedule(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr_ratio: float = 0.1):
    """Cosine learning rate schedule with linear warmup."""
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, val_loader, device, ctx):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with ctx:
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    model.train()

    # Average across ranks
    stats = torch.tensor([total_loss, total_tokens], device=device)
    dist.all_reduce(stats)
    return (stats[0] / stats[1]).item()


def train():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0

    # Seed
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    # Config
    cfg = SCALING_CONFIGS[args.config]
    max_steps = args.max_steps or cfg["max_steps"]
    warmup_steps = max(1, int(max_steps * 0.1))

    # Micro batch and grad accumulation
    global_batch = cfg["batch_size"]  # number of sequences
    if args.micro_batch is not None:
        micro_batch = args.micro_batch
    else:
        # Auto: full_attnres needs less memory due to O(L^2) saved activations
        if args.variant == "full_attnres":
            micro_batch = min(2, global_batch // world_size)
        elif args.variant == "block_attnres":
            micro_batch = min(4, global_batch // world_size)
        else:
            micro_batch = min(8, global_batch // world_size)
    grad_accum_steps = max(1, global_batch // (micro_batch * world_size))
    effective_batch = micro_batch * world_size * grad_accum_steps

    if is_master:
        print(f"Config: {args.config}, Variant: {args.variant}")
        print(f"Global batch: {effective_batch} seqs, Micro batch: {micro_batch}/GPU, "
              f"Grad accum: {grad_accum_steps}, World size: {world_size}")
        print(f"Max steps: {max_steps}, Warmup: {warmup_steps}")

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
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"FLOPs/token (fwd): {model.flops_per_token():,.0f}")

    if args.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # Mixed precision
    ctx = autocast(device_type="cuda", dtype=torch.bfloat16)
    scaler = GradScaler(enabled=False)  # bf16 doesn't need scaler

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=micro_batch,
        seq_len=SEQ_LEN,
        rank=rank,
        world_size=world_size,
    )

    # wandb
    use_wandb = is_master
    if is_master:
        run_name = f"{args.config}_{args.variant}"
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    "model_config": args.config,
                    "variant": args.variant,
                    "num_blocks": args.num_blocks,
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
                    "seq_len": SEQ_LEN,
                    "num_params": model.module.count_parameters() if hasattr(model, "module") else model.count_parameters(),
                },
            )
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}), continuing without wandb")
            use_wandb = False

    # Training loop
    model.train()
    optimizer.zero_grad()
    step = 0
    tokens_seen = 0
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
            # Sync gradients only on last micro step
            sync_context = model.no_sync if micro_step < grad_accum_steps - 1 else nullcontext
            with sync_context():
                with ctx:
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                    loss = loss / grad_accum_steps
                loss.backward()
            accum_loss += loss.item()
            tokens_seen += inputs.numel() * world_size

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += accum_loss
        step += 1

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = tokens_seen / elapsed
            avg_loss = running_loss / args.log_interval
            # Compute FLOPs
            raw_model = model.module if hasattr(model, "module") else model
            total_flops = 6 * raw_model.flops_per_token() * tokens_seen  # 6x for fwd+bwd
            pflops_days = total_flops / (1e15 * 86400)

            if is_master:
                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/tokens_seen": tokens_seen,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/pflops_days": pflops_days,
                        "train/step": step,
                    }, step=step)
                print(f"Step {step}/{max_steps} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"tok/s {tokens_per_sec:.0f} | PFLOP/s-days {pflops_days:.4f}")
            running_loss = 0.0

        # Validation
        if step % args.val_interval == 0 or step == max_steps:
            val_loss = validate(model, val_loader, device, ctx)
            raw_model = model.module if hasattr(model, "module") else model
            total_flops = 6 * raw_model.flops_per_token() * tokens_seen
            pflops_days = total_flops / (1e15 * 86400)
            if is_master:
                if use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/pflops_days": pflops_days,
                        "train/step": step,
                    }, step=step)
                print(f"  >> Val loss: {val_loss:.4f} at {pflops_days:.4f} PFLOP/s-days")

    # Save final results
    if is_master:
        raw_model = model.module if hasattr(model, "module") else model
        total_flops = 6 * raw_model.flops_per_token() * tokens_seen
        pflops_days = total_flops / (1e15 * 86400)

        results = {
            "config": args.config,
            "variant": args.variant,
            "val_loss": val_loss,
            "pflops_days": pflops_days,
            "tokens_seen": tokens_seen,
            "num_params": raw_model.count_parameters(),
        }
        os.makedirs(args.save_dir, exist_ok=True)
        result_file = os.path.join(args.save_dir, f"{args.config}_{args.variant}_results.pt")
        torch.save(results, result_file)
        print(f"\nFinal results saved to {result_file}")
        print(f"Config: {args.config}, Variant: {args.variant}")
        print(f"Val loss: {val_loss:.4f}, PFLOP/s-days: {pflops_days:.4f}")

        if use_wandb:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
