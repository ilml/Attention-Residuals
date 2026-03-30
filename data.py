"""
Data loading for Attention Residuals experiments.
Supports two modes:
  1. Small sample: loads parquets, tokenizes on the fly (for quick tests)
  2. Large binary: loads pre-tokenized mmap files from prepare_data.py (for real runs)
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

VOCAB_SIZE = 50257  # GPT-2
SEQ_LEN = 8192

# Default paths
SAMPLE_DATA_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/tolong/data/Nemotron-Pretraining-Dataset-sample"
LARGE_DATA_DIR = "/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized_150B"


class MmapTokenDataset(Dataset):
    """Memory-mapped dataset for pre-tokenized binary data."""

    def __init__(self, bin_path: str, n_seqs: int, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.int32, mode="r", shape=(n_seqs, seq_len))
        self.n_seqs = n_seqs

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.data[idx].astype(np.int64))
        return tokens[:-1], tokens[1:]


class InMemoryTokenDataset(Dataset):
    """In-memory dataset for small data (original sample)."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.data[idx].astype(np.int64))
        return tokens[:-1], tokens[1:]


def _load_sample_data(seq_len: int):
    """Load the small sample dataset (tokenize on the fly with caching)."""
    import tiktoken
    import pandas as pd

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tokens_seqlen{seq_len}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["train"], data["val"]

    # Load and tokenize
    texts = []
    for pf in sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, "*/*.parquet"))):
        try:
            df = pd.read_parquet(pf, columns=["text"])
            texts.extend(df["text"].dropna().tolist())
        except Exception:
            continue

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    all_tokens = []
    for text in texts:
        all_tokens.extend(enc.encode_ordinary(text))
        all_tokens.append(eot)

    n_seqs = len(all_tokens) // seq_len
    packed = np.array(all_tokens[:n_seqs * seq_len], dtype=np.int32).reshape(n_seqs, seq_len)

    n_val = max(1, int(n_seqs * 0.05))
    train, val = packed[:-n_val], packed[-n_val:]
    np.savez(cache_file, train=train, val=val)
    return train, val


def get_dataloaders(
    batch_size: int,
    seq_len: int = SEQ_LEN,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    data_dir: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val dataloaders.
    If data_dir points to a directory with meta.json (from prepare_data.py),
    uses mmap loading. Otherwise falls back to the small sample dataset.
    """
    if data_dir is None:
        data_dir = LARGE_DATA_DIR

    meta_path = os.path.join(data_dir, "meta.json")

    if os.path.exists(meta_path):
        # Large binary dataset
        with open(meta_path) as f:
            meta = json.load(f)
        if rank == 0:
            print(f"Using large dataset: {meta['n_train']:,} train + {meta['n_val']:,} val seqs "
                  f"({meta['total_tokens']/1e9:.1f}B tokens)")

        train_ds = MmapTokenDataset(
            os.path.join(data_dir, "train.bin"), meta["n_train"], meta["seq_len"])
        val_ds = MmapTokenDataset(
            os.path.join(data_dir, "val.bin"), meta["n_val"], meta["seq_len"])
    else:
        # Fallback to small sample
        if rank == 0:
            print(f"Large dataset not found at {data_dir}, using small sample")
        train_data, val_data = _load_sample_data(seq_len)
        train_ds = InMemoryTokenDataset(train_data)
        val_ds = InMemoryTokenDataset(val_data)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return train_loader, val_loader
