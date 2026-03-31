"""
Pre-tokenize Nemotron parquet files into binary format for fast training.
Uses plain file I/O (not mmap) for writing to avoid page cache OOM.

Usage:
  python prepare_data.py --data_dir /path/to/nemotron --out_dir /path/to/output \
      --max_tokens 150_000_000_000 --seq_len 8192
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
    now = time.time()
    files = []
    for pf in sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)):
        mtime = os.path.getmtime(pf)
        if (now - mtime) > min_age_seconds and os.path.getsize(pf) > 1_000_000:
            files.append(pf)
    return files


def _tokenize_chunk(texts: list[str]) -> list[int]:
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
    import pandas as pd
    df = pd.read_parquet(pf, columns=["text"])
    texts = df["text"].dropna().tolist()
    del df
    if not texts:
        return np.array([], dtype=np.int32)

    n_chunks = min(workers, max(1, len(texts) // 1000))
    if n_chunks <= 1:
        return np.array(_tokenize_chunk(texts), dtype=np.int32)

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
                        default="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized_150B")
    parser.add_argument("--max_tokens", type=int, default=150_000_000_000)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--val_fraction", type=float, default=0.002)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min_age", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Already done: {meta['n_train']+meta['n_val']:,} seqs, "
              f"{meta['total_tokens']/1e9:.1f}B tokens")
        return

    # Clean up partial runs
    for f in ["train.bin", "val.bin", "tokens_tmp.bin"]:
        p = os.path.join(args.out_dir, f)
        if os.path.exists(p):
            os.remove(p)

    print(f"Scanning {args.data_dir}...")
    parquet_files = get_safe_parquet_files(args.data_dir, args.min_age)
    print(f"Found {len(parquet_files)} safe parquet files")

    seq_len = args.seq_len
    max_seqs = args.max_tokens // seq_len
    n_val = max(1, int(max_seqs * args.val_fraction))
    n_train = max_seqs - n_val
    total_target = n_train + n_val

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    print(f"Target: {n_train:,} train + {n_val:,} val seqs = {total_target * seq_len / 1e9:.1f}B tokens")

    # Write directly to binary files, drop page cache after each file to avoid OOM
    f_train = open(train_path, "wb")
    f_val = open(val_path, "wb")

    seq_written = 0
    remainder = np.array([], dtype=np.int32)
    t0 = time.time()

    for i, pf in enumerate(parquet_files):
        try:
            tokens = tokenize_parquet(pf, workers=args.workers)
        except Exception as e:
            print(f"  Error on {pf}: {e}, skipping")
            continue

        # Prepend remainder from last file
        if len(remainder) > 0:
            tokens = np.concatenate([remainder, tokens])
            remainder = np.array([], dtype=np.int32)

        # Extract complete sequences
        n_complete = len(tokens) // seq_len
        if n_complete > 0:
            usable = tokens[:n_complete * seq_len]
            remainder = tokens[n_complete * seq_len:]
            del tokens

            # Write sequences to train or val file
            for j in range(n_complete):
                seq_bytes = usable[j * seq_len:(j + 1) * seq_len].tobytes()
                if seq_written < n_train:
                    f_train.write(seq_bytes)
                elif seq_written < total_target:
                    f_val.write(seq_bytes)
                else:
                    break
                seq_written += 1

            del usable
        else:
            remainder = tokens
            del tokens

        if seq_written >= total_target:
            print(f"  Reached target: {seq_written:,} sequences")
            break

        # Drop page cache to prevent OOM on large writes
        f_train.flush()
        f_val.flush()
        try:
            os.posix_fadvise(f_train.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(f_val.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass

        elapsed = time.time() - t0
        rate = (seq_written * seq_len) / elapsed if elapsed > 0 else 0
        basename = os.path.basename(os.path.dirname(pf)) + "/" + os.path.basename(pf)
        print(f"  [{i+1}/{len(parquet_files)}] {basename}: "
              f"{seq_written * seq_len / 1e9:.2f}B tokens | "
              f"{rate / 1e6:.1f}M tok/s | {elapsed:.0f}s | "
              f"{seq_written:,}/{total_target:,} seqs")

    f_train.close()
    f_val.close()

    actual_train = min(seq_written, n_train)
    actual_val = max(0, seq_written - n_train)
    total_written = (actual_train + actual_val) * seq_len

    meta = {
        "n_train": actual_train,
        "n_val": actual_val,
        "seq_len": seq_len,
        "vocab_size": 50257,
        "total_tokens": total_written,
        "num_parquet_files": i + 1,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/3600:.1f}h)!")
    print(f"  Train: {actual_train:,} seqs ({actual_train * seq_len / 1e9:.1f}B tok) -> {train_path}")
    print(f"  Val:   {actual_val:,} seqs ({actual_val * seq_len / 1e9:.1f}B tok) -> {val_path}")
    print(f"  Total: {total_written/1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
