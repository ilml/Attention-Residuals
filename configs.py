"""
Model configurations for scaling law experiments.

Two sets:
  SCALING_CONFIGS_SMALL  — for quick tests on the sample dataset (4 GPUs)
  SCALING_CONFIGS        — for proper runs on the full Nemotron dataset (32 GPUs)

Token budgets follow Chinchilla scaling (~20 tokens per parameter).
"""

# ---------------------------------------------------------------------------
# Large-scale configs for 32 GPUs (8 nodes x 4 GPUs)
# Chinchilla-optimal: ~20 tokens/param, proper batch sizes
# ---------------------------------------------------------------------------
SCALING_CONFIGS = {
    "124M": {
        "n_layer": 12,
        "d_model": 768,
        "n_head": 12,
        "d_ff": 2048,
        "lr": 3.0e-3,
        "batch_size": 192,    # sequences of 8192 tokens
        "total_tokens": 2_500_000_000,  # 2.5B tokens
    },
    "172M": {
        "n_layer": 13,
        "d_model": 896,
        "n_head": 14,
        "d_ff": 2432,
        "lr": 2.8e-3,
        "batch_size": 256,
        "total_tokens": 3_400_000_000,
    },
    "231M": {
        "n_layer": 14,
        "d_model": 1024,
        "n_head": 16,
        "d_ff": 2816,
        "lr": 2.5e-3,
        "batch_size": 320,
        "total_tokens": 4_600_000_000,
    },
    "313M": {
        "n_layer": 16,
        "d_model": 1152,
        "n_head": 18,
        "d_ff": 3072,
        "lr": 2.2e-3,
        "batch_size": 384,
        "total_tokens": 6_300_000_000,
    },
    "401M": {
        "n_layer": 17,
        "d_model": 1280,
        "n_head": 20,
        "d_ff": 3456,
        "lr": 2.0e-3,
        "batch_size": 448,
        "total_tokens": 8_000_000_000,
    },
}

# Compute max_steps from total_tokens for each config
for _name, _cfg in SCALING_CONFIGS.items():
    _tokens_per_step = _cfg["batch_size"] * 8192
    _cfg["max_steps"] = _cfg["total_tokens"] // _tokens_per_step


# ---------------------------------------------------------------------------
# Small configs for quick tests on the sample dataset (4 GPUs, 1 node)
# ---------------------------------------------------------------------------
SCALING_CONFIGS_SMALL = {
    "124M": {**SCALING_CONFIGS["124M"], "max_steps": 200, "batch_size": 192},
    "172M": {**SCALING_CONFIGS["172M"], "max_steps": 300, "batch_size": 256},
    "231M": {**SCALING_CONFIGS["231M"], "max_steps": 400, "batch_size": 320},
    "313M": {**SCALING_CONFIGS["313M"], "max_steps": 150, "batch_size": 384},
    "401M": {**SCALING_CONFIGS["401M"], "max_steps": 100, "batch_size": 448},
}
