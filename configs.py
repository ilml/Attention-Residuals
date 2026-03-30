"""
Model configurations for scaling law experiments.

SCALING_CONFIGS_PAPER: Exact reproduction of Table 2 from the paper.
  - Token counts match the paper exactly (38.7B - 119B)
  - Dense models approximate the paper's MoE activated parameter counts
  - d_model adjusted for clean MHA head dimensions (head_dim=64)
  - d_ff uses SwiGLU standard ratio: round(8/3 * d_model)

SCALING_CONFIGS_SMALL: Quick tests on sample dataset (4 GPUs, <1h)
"""

# ---------------------------------------------------------------------------
# Paper-exact configs (Table 2 reproduction)
# Dense equivalents of the MoE models, trained for the same token counts
# ---------------------------------------------------------------------------
SCALING_CONFIGS = {
    "194M": {
        "n_layer": 12,      # L_b=12 (paper)
        "d_model": 768,     # paper: 896, adjusted for head_dim=64
        "n_head": 12,       # paper: 12
        "d_ff": 2048,       # SwiGLU: round(8/3 * 768)
        "lr": 2.99e-3,      # paper exact
        "batch_size": 192,  # paper exact
        "total_tokens": 38_700_000_000,  # paper exact: 38.7B
    },
    "241M": {
        "n_layer": 13,
        "d_model": 896,     # paper: 960, adjusted for head_dim=64
        "n_head": 14,
        "d_ff": 2432,
        "lr": 2.80e-3,
        "batch_size": 256,
        "total_tokens": 45_400_000_000,  # paper exact: 45.4B
    },
    "296M": {
        "n_layer": 14,
        "d_model": 1024,    # paper: 1024
        "n_head": 16,
        "d_ff": 2816,
        "lr": 2.50e-3,
        "batch_size": 320,
        "total_tokens": 62_100_000_000,  # paper exact: 62.1B
    },
    "436M": {
        "n_layer": 16,
        "d_model": 1152,    # paper: 1168, adjusted for head_dim=64
        "n_head": 18,
        "d_ff": 3072,
        "lr": 2.20e-3,
        "batch_size": 384,
        "total_tokens": 87_900_000_000,  # paper exact: 87.9B
    },
    "528M": {
        "n_layer": 17,
        "d_model": 1280,    # paper: 1264, adjusted for head_dim=64
        "n_head": 20,
        "d_ff": 3456,
        "lr": 2.02e-3,
        "batch_size": 432,
        "total_tokens": 119_000_000_000, # paper exact: 119B
    },
}

# Compute max_steps from total_tokens
for _name, _cfg in SCALING_CONFIGS.items():
    _tokens_per_step = _cfg["batch_size"] * 8192
    _cfg["max_steps"] = _cfg["total_tokens"] // _tokens_per_step


# ---------------------------------------------------------------------------
# Small configs for quick tests
# ---------------------------------------------------------------------------
SCALING_CONFIGS_SMALL = {
    "194M": {**SCALING_CONFIGS["194M"], "max_steps": 200},
    "241M": {**SCALING_CONFIGS["241M"], "max_steps": 300},
    "296M": {**SCALING_CONFIGS["296M"], "max_steps": 400},
    "436M": {**SCALING_CONFIGS["436M"], "max_steps": 150},
    "528M": {**SCALING_CONFIGS["528M"], "max_steps": 100},
}
