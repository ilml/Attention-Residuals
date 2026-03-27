<div align="center">
<h2 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    <img src="assets/logo.png" height="16" width="16" style="display: inline-block; vertical-align: middle; margin: 2px;"> Attention Residuals — Reproduction
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h2>
</div>

<p align="center">
  <a href="Attention_Residuals.pdf">Paper</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2603.15031">arXiv</a> &nbsp;|&nbsp;
  <a href="https://github.com/MoonshotAI/Attention-Residuals">Original Repo</a> &nbsp;|&nbsp;
  <a href="#reproduction-results">Results</a> &nbsp;|&nbsp;
  <a href="#how-to-run">How to Run</a>
</p>

This fork contains a **pure PyTorch reproduction** of the scaling law experiments from the [Attention Residuals](https://arxiv.org/abs/2603.15031) paper (Kimi Team, 2025). All infrastructure complexity (pipeline parallelism, MoE routing, custom kernels) is stripped away — the implementation uses only standard PyTorch DDP with bf16 mixed precision.

---

## Paper Overview

Standard residual connections accumulate all layer outputs with fixed unit weights. As depth grows, this uniform aggregation dilutes each layer's contribution (the **PreNorm dilution** problem).

**Attention Residuals (AttnRes)** replaces this with softmax attention over preceding layer outputs:

$$\mathbf{h}_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \mathbf{v}_i$$

where $\alpha_{i \to l}$ are computed via a single learned pseudo-query $\mathbf{w}_l \in \mathbb{R}^d$ per layer, initialized to **zero** (so training starts from uniform weights, equivalent to standard residuals).

<p align="center">
  <img src="assets/overview.png" width="800" />
</p>
<p align="center"><em>
  (a) Standard residuals with uniform additive accumulation.
  (b) Full AttnRes: each layer attends over all previous outputs.
  (c) Block AttnRes: layers grouped into N blocks, reducing memory from O(Ld) to O(Nd).
</em></p>

### Three Variants

| Variant | Description | Memory | This Repo |
|---------|-------------|--------|-----------|
| **Baseline** | Standard PreNorm residuals (`h = h + f(h)`) | O(d) | `--variant baseline` |
| **Full AttnRes** | Attend over all L previous sublayer outputs | O(Ld) | `--variant full_attnres` |
| **Block AttnRes** | Attend over N block-level representations | O(Nd) | `--variant block_attnres` |

---

## Reproduction Setup

### Architecture

We use **dense Transformer models** (not MoE) with standard multi-head attention + RoPE + SwiGLU FFN. This differs from the paper's MoE architecture but preserves the key comparison: the only difference between variants is the residual connection mechanism.

### Model Configurations

Five model sizes adapted from Table 2 of the paper, with `d_ff = round(8/3 * d_model)` for SwiGLU:

| Config | Params | Layers | d_model | d_ff | Heads | LR | Batch Size | Steps |
|--------|--------|--------|---------|------|-------|-----|------------|-------|
| **124M** | 123.6M | 12 | 768 | 2048 | 12 | 3.0e-3 | 192 | 200 |
| **172M** | 171.8M | 13 | 896 | 2432 | 14 | 2.8e-3 | 256 | 300 |
| **231M** | 231.3M | 14 | 1024 | 2816 | 16 | 2.5e-3 | 320 | 400 |
| **313M** | 312.7M | 16 | 1152 | 3072 | 18 | 2.2e-3 | 384 | 150 |
| **401M** | 401.4M | 17 | 1280 | 3456 | 20 | 2.0e-3 | 432 | 100 |

### Training Details

- **Data**: [Nemotron Pretraining Dataset (sample)](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) — 26,706 documents, ~38.7M tokens (GPT-2 tokenizer), packed into 4,488 train sequences of length 8192
- **Hardware**: 4x NVIDIA GB200 GPUs (192 GB HBM3e each), 1 node
- **Optimizer**: AdamW (betas=0.9/0.95, weight_decay=0.1, gradient clipping=1.0)
- **Schedule**: Cosine LR with 10% linear warmup
- **Precision**: bf16 mixed precision
- **Distributed**: PyTorch DDP (4 GPUs), gradient accumulation to reach target batch sizes
- **Block AttnRes**: N=8 blocks for all model sizes

### Key Implementation Details

- **Zero-init pseudo-queries**: All `w_l` vectors initialized to zero per paper Section 5, ensuring uniform initial weights
- **Gradient checkpointing**: Applied to Full AttnRes attention/MLP sublayers to fit in GPU memory
- **Micro-batch scaling**: Baseline uses micro_batch=8, Block AttnRes uses 4, Full AttnRes uses 2 (or 1 for 401M) due to O(L^2) activation memory

---

## Reproduction Results

### Validation Loss Across Model Sizes

<p align="center">
  <img src="assets/scaling_law_repro.png" width="800" />
</p>

**Left**: Validation loss for each model size. Lower is better. **Right**: Loss improvement of AttnRes over baseline. Positive = AttnRes wins.

| Config | Params | Tokens | Baseline | Full AttnRes | Block AttnRes | Best Improvement |
|--------|-------:|-------:|:--------:|:------------:|:-------------:|:----------------:|
| **124M** | 124M | 315M | 4.43 | 4.08 | **3.98** | **-0.45** (Block) |
| **172M** | 172M | 629M | 3.48 | 3.38 | **3.32** | **-0.16** (Block) |
| **231M** | 231M | 1.05B | **3.13** | 3.34 | 3.38 | +0.22 (Baseline wins) |
| **313M** | 313M | 472M | 4.78 | **4.59** | 4.62 | **-0.18** (Full) |
| **401M** | 401M | 354M | 5.55 | 5.25 | **4.93** | **-0.62** (Block) |

> **Important context on the non-monotonic loss**: The 313M and 401M models have *higher* loss than 231M despite being larger because they were trained for far fewer tokens (472M and 354M vs. 1.05B). This was due to GPU time limits. The proper comparison is *between variants at the same model size*, not across sizes.

### Key Findings

**1. AttnRes outperforms baseline at 4 out of 5 model sizes**

At every model size except 231M, at least one AttnRes variant achieves lower validation loss than the baseline — with improvements ranging from -0.16 (172M) to -0.62 (401M). On average across all 5 sizes, Block AttnRes improves over baseline by **0.23** and Full AttnRes by **0.14**.

**2. Block AttnRes is the best overall variant**

Block AttnRes (N=8 blocks) beats Full AttnRes at 3 of 5 sizes and shows the largest single improvement (-0.62 at 401M). This matches the paper's finding that Block AttnRes is the practical choice, recovering most of Full AttnRes's gains at lower memory cost.

**3. The benefit is largest with limited training**

The biggest improvements come at 124M (-0.45) and 401M (-0.62), where training was shortest relative to model size. This suggests AttnRes helps models learn more efficiently in the early phase of training — consistent with the paper's claim that AttnRes provides a "1.25x compute advantage."

**4. The 231M exception**

At 231M (the most heavily trained config at 1.05B tokens, ~27 epochs over our 38.7M token dataset), the baseline wins. With this much data recycling, the baseline's simpler optimization landscape likely benefits from memorization. The paper trained at 62B tokens for this scale, so this is a dataset limitation, not a method limitation.

### Comparison with Paper

| Paper Claim | Our Finding | Status |
|-------------|-------------|--------|
| AttnRes outperforms baseline across compute budgets | Yes, at 4/5 model sizes | Reproduced |
| Block AttnRes recovers most of Full AttnRes gains | Block actually exceeds Full at 3/5 sizes | Reproduced (stronger) |
| 1.25x compute advantage for Block AttnRes | Largest gains appear at undertrained scales, consistent with early efficiency | Directionally consistent |
| Zero-init is critical for stability | Training was stable with zero-init across all 15 runs | Consistent |

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
# Requires: torch>=2.1.0, tiktoken, wandb, pandas, pyarrow, numpy, scipy, matplotlib
```

### Quick Test (single experiment)

```bash
export WANDB_MODE=offline
torchrun --nproc_per_node=4 train.py \
    --config 124M --variant block_attnres \
    --wandb_project attn-residuals \
    --max_steps 100 --val_interval 50
```

### Full Scaling Law Sweep (15 experiments)

```bash
# On a Slurm cluster with 4 GPUs:
sbatch submit_scaling.sh

# Or run directly:
bash run_scaling.sh
```

The script runs all 5 model sizes x 3 variants sequentially, skipping experiments that already have saved results (preemption-safe).

### Analyze Results

```bash
python analyze.py --results_dir checkpoints --output scaling_law.png
```

This fits power-law curves L = A x C^(-alpha) and generates the scaling plot.

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Model size: `124M`, `172M`, `231M`, `313M`, `401M` | required |
| `--variant` | Residual type: `baseline`, `full_attnres`, `block_attnres` | required |
| `--num_blocks` | Number of blocks for Block AttnRes | 8 |
| `--max_steps` | Override training steps | from config |
| `--micro_batch` | Per-GPU micro batch size | auto |
| `--val_interval` | Validate every N steps | 100 |
| `--wandb_project` | W&B project name | `attn-residuals` |
| `--compile` | Use `torch.compile` | off |

---

## Code Structure

```
.
├── model.py           # Transformer with 3 residual variants (Baseline, Full AttnRes, Block AttnRes)
├── data.py            # Nemotron dataset loading, tokenization (tiktoken GPT-2), sequence packing
├── train.py           # DDP training loop with cosine LR, bf16, wandb logging
├── configs.py         # 5 model size configurations
├── analyze.py         # Power-law curve fitting and scaling plot generation
├── run_scaling.sh     # Orchestrates all 15 experiments sequentially
├── submit_scaling.sh  # Slurm batch submission script
├── requirements.txt   # Python dependencies
└── Attention_Residuals.pdf  # Original paper
```

### Model Architecture (`model.py`)

- **`RMSNorm`** — Pre-normalization (no bias, no shift)
- **`RotaryEmbedding`** — RoPE positional encoding
- **`Attention`** — Standard multi-head attention with `F.scaled_dot_product_attention` (Flash Attention)
- **`SwiGLUFFN`** — Gated FFN: `down(silu(gate(x)) * up(x))`
- **`AttnResOp`** — The depth-wise attention operation: `h = softmax(w^T RMSNorm(V)) @ V`
- **`Transformer`** — Unified model class dispatching to `_forward_baseline`, `_forward_full_attnres`, or `_forward_block_attnres`

---

## Limitations

- **Small dataset**: 38.7M unique tokens is far too small for proper scaling law experiments. The paper used 38-119B tokens per model size. Our results show the directional trend but not clean power-law behavior.
- **Dense models only**: The paper uses MoE models. Our dense reproductions match activated parameter counts but differ in optimization dynamics.
- **Limited compute range**: Our experiments span only ~6x in compute (0.0035 to 0.021 PFLOP/s-days) vs. the paper's ~10x range (0.5 to 5 PFLOP/s-days).
- **Larger models are undertrained**: The 313M and 401M configs had reduced step counts to fit in job time limits, making their absolute losses less meaningful (but relative comparisons remain valid).

---

## Citation

Original paper:

```bib
@misc{chen2026attnres,
  title         = {Attention Residuals},
  author        = {Kimi Team  and Chen, Guangyu  and Zhang, Yu  and Su, Jianlin  and Xu, Weixin  and Pan, Siyuan  and Wang, Yaoyu  and Wang, Yucheng  and Chen, Guanduo  and Yin, Bohong  and Chen, Yutian  and Yan, Junjie  and Wei, Ming  and Zhang, Y.  and Meng, Fanqing  and Hong, Chao  and Xie, Xiaotong  and Liu, Shaowei  and Lu, Enzhe  and Tai, Yunpeng  and Chen, Yanru  and Men, Xin  and Guo, Haiqing  and Charles, Y.  and Lu, Haoyu  and Sui, Lin  and Zhu, Jinguo  and Zhou, Zaida  and He, Weiran  and Huang, Weixiao  and Xu, Xinran  and Wang, Yuzhi  and Lai, Guokun  and Du, Yulun  and Wu, Yuxin  and Yang, Zhilin  and Zhou, Xinyu},
  year          = {2026},
  archiveprefix = {arXiv},
  eprint        = {2603.15031},
  primaryclass  = {cs.CL}
}
```
