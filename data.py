"""
Data loading for the Nemotron-Pretraining-Dataset-sample.
Loads parquet files, tokenizes with tiktoken GPT-2, packs into fixed-length sequences.
"""

import os
import glob
import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

DATA_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/tolong/data/Nemotron-Pretraining-Dataset-sample"
VOCAB_SIZE = 50257  # GPT-2
SEQ_LEN = 8192


def load_texts(data_dir: str = DATA_DIR) -> list[str]:
    """Load all text from parquet files in the dataset directory."""
    import pandas as pd

    texts = []
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*/*.parquet")))
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            print(f"Warning: skipping {pf}: {e}")
            continue
        if "text" in df.columns:
            texts.extend(df["text"].dropna().tolist())
        else:
            print(f"Skipping {pf} (no 'text' column, has: {list(df.columns)})")
    print(f"Loaded {len(texts)} documents from {len(parquet_files)} parquet files")
    return texts


def tokenize_and_pack(
    texts: list[str],
    seq_len: int = SEQ_LEN,
    val_fraction: float = 0.05,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tokenize all texts and pack into fixed-length sequences.
    Returns (train_tokens, val_tokens) as 2D numpy arrays of shape [N, seq_len].
    """
    # Check for cached tokenized data
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tokens_seqlen{seq_len}.npz")

    if os.path.exists(cache_file):
        print(f"Loading cached tokenized data from {cache_file}")
        data = np.load(cache_file)
        return data["train"], data["val"]

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # <|endoftext|> separator

    # Tokenize all documents
    print("Tokenizing documents...")
    all_tokens = []
    for i, text in enumerate(texts):
        tokens = enc.encode_ordinary(text)
        all_tokens.extend(tokens)
        all_tokens.append(eot)
        if (i + 1) % 5000 == 0:
            print(f"  Tokenized {i+1}/{len(texts)} docs, {len(all_tokens)} tokens so far")

    total = len(all_tokens)
    print(f"Total tokens: {total:,}")

    # Pack into sequences of seq_len
    n_seqs = total // seq_len
    packed = np.array(all_tokens[: n_seqs * seq_len], dtype=np.int32).reshape(n_seqs, seq_len)
    print(f"Packed into {n_seqs} sequences of length {seq_len}")

    # Split train / val
    n_val = max(1, int(n_seqs * val_fraction))
    n_train = n_seqs - n_val
    # Deterministic split: last n_val sequences are validation
    train = packed[:n_train]
    val = packed[n_train:]

    print(f"Train: {n_train} seqs ({n_train * seq_len:,} tokens)")
    print(f"Val:   {n_val} seqs ({n_val * seq_len:,} tokens)")

    np.savez(cache_file, train=train, val=val)
    print(f"Saved cache to {cache_file}")
    return train, val


class TokenDataset(Dataset):
    """Simple dataset that wraps packed token sequences."""
    def __init__(self, data: np.ndarray):
        self.data = data  # [N, seq_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.data[idx].astype(np.int64))
        # For language modeling: input = tokens[:-1], target = tokens[1:]
        return tokens[:-1], tokens[1:]


def get_dataloaders(
    batch_size: int,
    seq_len: int = SEQ_LEN,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders with DistributedSampler."""
    texts = load_texts()
    train_data, val_data = tokenize_and_pack(texts, seq_len=seq_len)

    train_ds = TokenDataset(train_data)
    val_ds = TokenDataset(val_data)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    texts = load_texts()
    train, val = tokenize_and_pack(texts)
    print(f"Train shape: {train.shape}, Val shape: {val.shape}")
