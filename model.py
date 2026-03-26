"""
Transformer model with three residual connection variants:
  1. Baseline (standard PreNorm residuals)
  2. Full Attention Residuals (AttnRes)
  3. Block Attention Residuals (Block AttnRes)

Reference: "Attention Residuals" (Kimi Team, 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos, sin: [T, D] -> [1, 1, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# AttnRes operation  (Eq. 2-4 in paper)
# ---------------------------------------------------------------------------

class AttnResOp(nn.Module):
    """
    Depth-wise softmax attention: h = sum_i alpha_i * v_i
    where alpha_i = softmax(w^T RMSNorm(v_i)).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        # Pseudo-query w_l -- MUST be initialized to zero (paper Sec 5)
        self.w = nn.Parameter(torch.zeros(d_model))

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """
        sources: list of [B, T, D] tensors (layer outputs / block reps).
        Returns:  [B, T, D]
        """
        V = torch.stack(sources, dim=0)          # [N, B, T, D]
        K = self.norm(V)                          # [N, B, T, D]
        logits = torch.einsum("d, n b t d -> n b t", self.w, K)
        weights = logits.softmax(dim=0)           # [N, B, T]
        h = torch.einsum("n b t, n b t d -> b t d", weights, V)
        return h


# ---------------------------------------------------------------------------
# Transformer with configurable residual mode
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Decoder-only Transformer supporting three residual modes:
      - "baseline":       standard PreNorm residual connections
      - "full_attnres":   Full Attention Residuals (attend over every sublayer output)
      - "block_attnres":  Block Attention Residuals (attend over block representations)

    Architecture per transformer block: RMSNorm -> MHA -> (residual) -> RMSNorm -> SwiGLU -> (residual)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,       # number of transformer blocks (L_b in paper; L = 2 * n_layer sublayers)
        n_head: int,
        d_ff: int,
        max_seq_len: int = 8192,
        residual_mode: str = "baseline",
        num_blocks: int = 8,   # N for Block AttnRes
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.residual_mode = residual_mode
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size

        # Token embedding (tied with output head)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_head, max_seq_len)

        # Transformer layers
        self.attn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layer)])
        self.attns = nn.ModuleList([Attention(d_model, n_head) for _ in range(n_layer)])
        self.mlp_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layer)])
        self.mlps = nn.ModuleList([SwiGLUFFN(d_model, d_ff) for _ in range(n_layer)])

        # Final norm
        self.final_norm = RMSNorm(d_model)

        # Output head (weight-tied with embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # --- AttnRes-specific parameters ---
        num_sublayers = 2 * n_layer  # attn + mlp counted separately

        if residual_mode == "full_attnres":
            # One AttnResOp per sublayer + one for final output
            self.attn_res_ops = nn.ModuleList(
                [AttnResOp(d_model) for _ in range(num_sublayers + 1)]
            )

        elif residual_mode == "block_attnres":
            # Compute block assignments for sublayers
            self._sublayers_per_block = self._compute_block_sizes(num_sublayers, num_blocks)
            self._sublayer_to_block = []
            for blk_idx, size in enumerate(self._sublayers_per_block):
                self._sublayer_to_block.extend([blk_idx] * size)

            # One AttnResOp per sublayer + one for final output
            self.attn_res_ops = nn.ModuleList(
                [AttnResOp(d_model) for _ in range(num_sublayers + 1)]
            )

        self._init_weights()

    @staticmethod
    def _compute_block_sizes(num_sublayers: int, num_blocks: int) -> list[int]:
        """Distribute sublayers as evenly as possible among blocks."""
        base = num_sublayers // num_blocks
        extra = num_sublayers % num_blocks
        sizes = []
        for i in range(num_blocks):
            sizes.append(base + (1 if i < extra else 0))
        return sizes

    def _init_weights(self):
        """Standard initialization for Transformer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # AttnRes pseudo-queries are already zero-initialized in AttnResOp.__init__

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] token ids
        Returns:   [B, T, vocab_size] logits
        """
        if self.residual_mode == "baseline":
            return self._forward_baseline(input_ids)
        elif self.residual_mode == "full_attnres":
            return self._forward_full_attnres(input_ids)
        elif self.residual_mode == "block_attnres":
            return self._forward_block_attnres(input_ids)
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

    # -----------------------------------------------------------------------
    # Baseline: standard PreNorm residual
    # -----------------------------------------------------------------------
    def _forward_baseline(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed(input_ids)
        cos, sin = self.rope(T)

        for i in range(self.n_layer):
            h = h + self.attns[i](self.attn_norms[i](h), cos, sin)
            h = h + self.mlps[i](self.mlp_norms[i](h))

        h = self.final_norm(h)
        return self.lm_head(h)

    # -----------------------------------------------------------------------
    # Full AttnRes  (Sec 3.1)
    # -----------------------------------------------------------------------
    def _forward_full_attnres(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed(input_ids)
        cos, sin = self.rope(T)

        # sources[0] = embedding (v_0 = h_1)
        sources: list[torch.Tensor] = [h]

        op_idx = 0
        for i in range(self.n_layer):
            # Pre-attention: aggregate all sources
            h = self.attn_res_ops[op_idx](sources)
            # Use gradient checkpointing for attention to save memory
            attn_out = checkpoint(
                self.attns[i], self.attn_norms[i](h), cos, sin,
                use_reentrant=False,
            )
            sources.append(attn_out)
            op_idx += 1

            # Pre-MLP: aggregate all sources
            h = self.attn_res_ops[op_idx](sources)
            mlp_out = checkpoint(
                self.mlps[i], self.mlp_norms[i](h),
                use_reentrant=False,
            )
            sources.append(mlp_out)
            op_idx += 1

        # Final aggregation
        h = self.attn_res_ops[op_idx](sources)
        h = self.final_norm(h)
        return self.lm_head(h)

    # -----------------------------------------------------------------------
    # Block AttnRes  (Sec 3.2, Fig 2)
    # -----------------------------------------------------------------------
    def _forward_block_attnres(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed(input_ids)
        cos, sin = self.rope(T)

        # b_0 = embedding
        completed_blocks: list[torch.Tensor] = [h]
        partial: Optional[torch.Tensor] = None
        current_block = 0
        sublayer_in_block = 0

        op_idx = 0
        for i in range(self.n_layer):
            # --- Sublayer: attention ---
            # Check block boundary
            if sublayer_in_block >= self._sublayers_per_block[current_block]:
                if partial is not None:
                    completed_blocks.append(partial)
                    partial = None
                current_block += 1
                sublayer_in_block = 0

            # Build source list
            if partial is not None:
                sources = completed_blocks + [partial]
            else:
                sources = completed_blocks
            h = self.attn_res_ops[op_idx](sources)
            attn_out = self.attns[i](self.attn_norms[i](h), cos, sin)
            partial = partial + attn_out if partial is not None else attn_out
            sublayer_in_block += 1
            op_idx += 1

            # --- Sublayer: MLP ---
            if sublayer_in_block >= self._sublayers_per_block[current_block]:
                if partial is not None:
                    completed_blocks.append(partial)
                    partial = None
                current_block += 1
                sublayer_in_block = 0

            if partial is not None:
                sources = completed_blocks + [partial]
            else:
                sources = completed_blocks
            h = self.attn_res_ops[op_idx](sources)
            mlp_out = self.mlps[i](self.mlp_norms[i](h))
            partial = partial + mlp_out if partial is not None else mlp_out
            sublayer_in_block += 1
            op_idx += 1

        # Complete last block if partial remains
        if partial is not None:
            completed_blocks.append(partial)

        # Final aggregation over all completed blocks
        h = self.attn_res_ops[op_idx](completed_blocks)
        h = self.final_norm(h)
        return self.lm_head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flops_per_token(self) -> float:
        """Approximate FLOPs per token (forward pass only)."""
        d = self.d_model
        L = self.n_layer
        V = self.vocab_size
        d_ff = self.mlps[0].gate_proj.out_features
        # Attention: 4*d^2 (QKV + O) + 2*T*d (ignored, depends on T)
        # MLP: 3*d*d_ff (gate + up + down)
        # Embedding/output: 2*d*V
        flops_attn = L * 4 * d * d
        flops_mlp = L * 3 * d * d_ff
        flops_embed = 2 * d * V
        return float(flops_attn + flops_mlp + flops_embed)
