"""
Pre-tokenize Nemotron parquet files into binary mmap format for fast training.

Processes files ONE AT A TIME to control memory, using multiprocessing
only within each file for tokenization.

Usage:
  python prepare_data.py --data_dir /path/to/nemotron --out_dir /path/to/output \
      --max_tokens 12_000_000_000 --seq_len 8192
"""

import os
import sys
import json
import glob
import time
import argparse
import multiprocessing as mp

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
        if age > min_age_seconds and size > 1_000_000:
            files.append(pf)
    return files


def _tokenize_chunk(texts: list[str]) -> list[int]:
    """Worker function: tokenize a chunk of texts."""
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    tokens = []
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        tokens.extend(enc.encode_ordinary(text))
        tokens.append(eot)
    return tokens


def tokenize_parquet(pf: str, workers: int = 8) -> np.ndarray:
    """Read one parquet file, tokenize in parallel, return int32 array."""
    import pandas as pd
    df = pd.read_parquet(pf, columns=["text"])
    texts = df["text"].dropna().tolist()
    del df

    if not texts:
        return np.array([], dtype=np.int32)

    # Split texts into chunks for parallel tokenization
    n_chunks = min(workers, max(1, len(texts) // 1000))
    if n_chunks <= 1:
        tokens = _tokenize_chunk(texts)
        return np.array(tokens, dtype=np.int32)

    chunk_size = len(texts) // n_chunks
    chunks = [texts[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    if len(texts) % n_chunks:
        chunks[-1].extend(texts[n_chunks * chunk_size:])
    del texts

    with mp.Pool(n_chunks) as pool:
        results = pool.map(_tokenize_chunk, chunks)

    all_tokens = []
    for r in results:
        all_tokens.extend(r)
    return np.array(all_tokens, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron")
    parser.add_argument("--out_dir", type=str,
                        default="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized")
    parser.add_argument("--max_tokens", type=int, default=12_000_000_000)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--val_fraction", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for tokenization within each file")
    parser.add_argument("--min_age", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Already done: {meta['n_train']+meta['n_val']:,} seqs, "
              f"{meta['total_tokens']/1e9:.1f}B tokens")
        return

    # Temp file to accumulate tokens on disk
    tmp_path = os.path.join(args.out_dir, "tokens_tmp.bin")

    print(f"Scanning {args.data_dir}...")
    parquet_files = get_safe_parquet_files(args.data_dir, args.min_age)
    print(f"Found {len(parquet_files)} safe parquet files")

    total_tokens = 0
    t0 = time.time()

    # Open a temp file and write tokens incrementally
    with open(tmp_path, "wb") as f_out:
        for i, pf in enumerate(parquet_files):
            try:
                tokens = tokenize_parquet(pf, workers=args.workers)
            except Exception as e:
                print(f"  Error on {pf}: {e}, skipping")
                continue

            f_out.write(tokens.tobytes())
            total_tokens += len(tokens)
            del tokens

            elapsed = time.time() - t0
            rate = total_tokens / elapsed if elapsed > 0 else 0
            basename = os.path.basename(os.path.dirname(pf)) + "/" + os.path.basename(pf)
            print(f"  [{i+1}/{len(parquet_files)}] {basename}: "
                  f"{total_tokens/1e9:.2f}B tokens | {rate/1e6:.1f}M tok/s | {elapsed:.0f}s")

            if total_tokens >= args.max_tokens:
                print(f"  Reached {args.max_tokens/1e9:.0f}B target, stopping.")
                break

    print(f"\nTotal tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")

    # Load from temp file, pack into sequences, split, and save as mmap
    print("Packing into sequences...")
    seq_len = args.seq_len
    raw = np.memmap(tmp_path, dtype=np.int32, mode="r")
    n_usable = min(len(raw), args.max_tokens)
    n_seqs = n_usable // seq_len

    n_val = max(1, int(n_seqs * args.val_fraction))
    n_train = n_seqs - n_val
    print(f"  {n_seqs:,} sequences -> {n_train:,} train + {n_val:,} val")

    # Write train mmap
    train_path = os.path.join(args.out_dir, "train.bin")
    train_mm = np.memmap(train_path, dtype=np.int32, mode="w+", shape=(n_train, seq_len))
    for j in range(n_train):
        start = j * seq_len
        train_mm[j] = raw[start:start + seq_len]
    train_mm.flush()
    del train_mm
    print(f"  Wrote {train_path}")

    # Write val mmap
    val_path = os.path.join(args.out_dir, "val.bin")
    val_mm = np.memmap(val_path, dtype=np.int32, mode="w+", shape=(n_val, seq_len))
    for j in range(n_val):
        start = (n_train + j) * seq_len
        val_mm[j] = raw[start:start + seq_len]
    val_mm.flush()
    del val_mm, raw
    print(f"  Wrote {val_path}")

    # Cleanup temp file
    os.remove(tmp_path)

    # Save metadata
    meta = {
        "n_train": n_train, "n_val": n_val, "seq_len": seq_len,
        "vocab_size": 50257, "total_tokens": total_tokens,
        "num_parquet_files": min(i + 1, len(parquet_files)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)!")
    print(f"  Train: {n_train:,} seqs -> {train_path}")
    print(f"  Val:   {n_val:,} seqs -> {val_path}")


if __name__ == "__main__":
    main()
