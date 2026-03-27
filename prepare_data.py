"""
Pre-tokenize Nemotron parquet files into a binary mmap format for fast training.

Reads parquet files from completed directories, tokenizes with tiktoken GPT-2,
packs into fixed-length sequences, and saves as numpy memmap files.

Usage:
  python prepare_data.py --data_dir /path/to/nemotron --out_dir /path/to/output \
      --max_tokens 12_000_000_000 --seq_len 8192 --workers 32
"""

import os
import sys
import json
import glob
import time
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import tiktoken


def get_safe_parquet_files(data_dir: str, min_age_seconds: int = 300) -> list[str]:
    """Get parquet files that are likely complete (not being downloaded)."""
    now = time.time()
    files = []
    for pf in sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)):
        mtime = os.path.getmtime(pf)
        age = now - mtime
        size = os.path.getsize(pf)
        if age > min_age_seconds and size > 1_000_000:  # >1MB and >5min old
            files.append(pf)
        else:
            print(f"  Skipping (recent/small): {pf} (age={age:.0f}s, size={size/1e6:.1f}MB)")
    return files


def tokenize_texts(texts: list[str]) -> list[int]:
    """Tokenize a batch of texts, joining with EOT tokens."""
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    tokens = []
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        tokens.extend(enc.encode_ordinary(text))
        tokens.append(eot)
    return tokens


def process_parquet_file(pf: str) -> list[int]:
    """Read a parquet file and return tokenized content."""
    import pandas as pd
    try:
        df = pd.read_parquet(pf, columns=["text"])
        texts = df["text"].dropna().tolist()
        tokens = tokenize_texts(texts)
        return tokens
    except Exception as e:
        print(f"  Error processing {pf}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize Nemotron data for training")
    parser.add_argument("--data_dir", type=str,
                        default="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron")
    parser.add_argument("--out_dir", type=str, default="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized")
    parser.add_argument("--max_tokens", type=int, default=12_000_000_000,
                        help="Stop after this many tokens")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--val_fraction", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--min_age", type=int, default=300,
                        help="Min file age in seconds to consider complete")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Check if already done
    meta_path = os.path.join(args.out_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Data already prepared: {meta['n_train']} train + {meta['n_val']} val sequences")
        print(f"Total tokens: {(meta['n_train'] + meta['n_val']) * meta['seq_len']:,}")
        return

    print(f"Scanning {args.data_dir} for parquet files...")
    parquet_files = get_safe_parquet_files(args.data_dir, min_age_seconds=args.min_age)
    print(f"Found {len(parquet_files)} safe parquet files")

    if not parquet_files:
        print("No files found! Check data_dir and min_age settings.")
        sys.exit(1)

    # Process files in parallel, collecting tokens
    print(f"\nTokenizing with {args.workers} workers...")
    all_tokens = []
    total_tokens = 0
    t0 = time.time()

    # Process in batches to control memory and allow early stopping
    batch_size = args.workers * 2
    for batch_start in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[batch_start:batch_start + batch_size]

        with mp.Pool(min(args.workers, len(batch_files))) as pool:
            results = pool.map(process_parquet_file, batch_files)

        for tokens in results:
            all_tokens.extend(tokens)
            total_tokens = len(all_tokens)

        elapsed = time.time() - t0
        rate = total_tokens / elapsed if elapsed > 0 else 0
        print(f"  Files {batch_start+len(batch_files)}/{len(parquet_files)} | "
              f"{total_tokens/1e9:.2f}B tokens | {rate/1e6:.1f}M tok/s | "
              f"{elapsed:.0f}s elapsed")

        if total_tokens >= args.max_tokens:
            print(f"  Reached target of {args.max_tokens/1e9:.1f}B tokens, stopping.")
            all_tokens = all_tokens[:args.max_tokens]
            break

    total_tokens = len(all_tokens)
    print(f"\nTotal tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")

    # Pack into sequences
    seq_len = args.seq_len
    n_seqs = total_tokens // seq_len
    print(f"Packing into {n_seqs:,} sequences of length {seq_len}")

    packed = np.array(all_tokens[:n_seqs * seq_len], dtype=np.int32).reshape(n_seqs, seq_len)
    del all_tokens  # Free memory

    # Split train / val
    n_val = max(1, int(n_seqs * args.val_fraction))
    n_train = n_seqs - n_val

    # Shuffle before split for good distribution
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_seqs)
    train_indices = np.sort(indices[:n_train])
    val_indices = np.sort(indices[n_train:])

    # Save as memmap files
    print(f"\nSaving train ({n_train:,} seqs) and val ({n_val:,} seqs)...")

    train_path = os.path.join(args.out_dir, "train.bin")
    train_mmap = np.memmap(train_path, dtype=np.int32, mode="w+", shape=(n_train, seq_len))
    train_mmap[:] = packed[train_indices]
    train_mmap.flush()

    val_path = os.path.join(args.out_dir, "val.bin")
    val_mmap = np.memmap(val_path, dtype=np.int32, mode="w+", shape=(n_val, seq_len))
    val_mmap[:] = packed[val_indices]
    val_mmap.flush()

    del packed, train_mmap, val_mmap

    # Save metadata
    meta = {
        "n_train": n_train,
        "n_val": n_val,
        "seq_len": seq_len,
        "vocab_size": 50257,
        "total_tokens": total_tokens,
        "num_parquet_files": len(parquet_files),
        "val_fraction": args.val_fraction,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s!")
    print(f"  Train: {n_train:,} seqs ({n_train * seq_len:,} tokens) -> {train_path}")
    print(f"  Val:   {n_val:,} seqs ({n_val * seq_len:,} tokens) -> {val_path}")
    print(f"  Meta:  {meta_path}")


if __name__ == "__main__":
    main()
