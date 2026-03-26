"""
Model configurations for the scaling law experiments.

Five model sizes following Table 2 of the Attention Residuals paper,
adapted for dense (non-MoE) models with standard MHA.

d_ff = round(8/3 * d_model) for SwiGLU, rounded to nearest 64.
Step counts chosen to give a ~20x compute range across configs.
"""

SCALING_CONFIGS = {
    "124M": {
        "n_layer": 12,
        "d_model": 768,
        "n_head": 12,
        "d_ff": 2048,
        "lr": 3.0e-3,
        "batch_size": 192,
        "max_steps": 200,
    },
    "172M": {
        "n_layer": 13,
        "d_model": 896,
        "n_head": 14,
        "d_ff": 2432,
        "lr": 2.8e-3,
        "batch_size": 256,
        "max_steps": 300,
    },
    "231M": {
        "n_layer": 14,
        "d_model": 1024,
        "n_head": 16,
        "d_ff": 2816,
        "lr": 2.5e-3,
        "batch_size": 320,
        "max_steps": 400,
    },
    "313M": {
        "n_layer": 16,
        "d_model": 1152,
        "n_head": 18,
        "d_ff": 3072,
        "lr": 2.2e-3,
        "batch_size": 384,
        "max_steps": 500,
    },
    "401M": {
        "n_layer": 17,
        "d_model": 1280,
        "n_head": 20,
        "d_ff": 3456,
        "lr": 2.0e-3,
        "batch_size": 432,
        "max_steps": 600,
    },
}
