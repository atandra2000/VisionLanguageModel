import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class LanguageConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024          # Reduced from 2048
    intermediate_size: int = 2048    # Reduced from 8192
    num_hidden_layers: int = 12      # Reduced from 18
    num_attention_heads: int = 8     # Reduced from 16
    num_key_value_heads: int = 4     # Grouped Query Attention
    head_dim: int = 128
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation for improved training stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for length-generalising positional encoding."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) — multiple query heads share a smaller set of
    key/value heads, reducing KV-cache memory during inference.
    """

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        hidden = config.hidden_size

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.attn_dropout = config.attention_probs_dropout_prob

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_position_embeddings, config.max_position_embeddings), diagonal=1).bool(),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:T, :T].to(attn.device), float("-inf"))
        if attention_mask is not None:
            attn = attn + attention_mask[:, None, None, :].to(attn.device)
        attn = F.dropout(F.softmax(attn, dim=-1), p=self.attn_dropout, training=self.training)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(out), new_kv_cache


class GeGLU(nn.Module):
    """Gated Linear Unit with GELU — provides richer non-linearity vs standard FFN."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    """Single Gemma-style decoder layer: pre-norm GQA + pre-norm GeGLU FFN."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = GeGLU(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.self_attn(self.input_layernorm(hidden_states), attention_mask, kv_cache)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, new_kv


class GemmaLanguageModel(nn.Module):
    """
    Gemma-style language model: token embeddings → decoder stack → RMSNorm → LM head.
    Embedding weights are tied to the LM head for parameter efficiency.
    """

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        old_n = self.config.vocab_size
        self.config.vocab_size = new_num_tokens
        old_emb, old_head = self.embed_tokens, self.lm_head

        self.embed_tokens = nn.Embedding(new_num_tokens, self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, new_num_tokens, bias=False)

        copy_n = min(old_n, new_num_tokens)
        with torch.no_grad():
            self.embed_tokens.weight[:copy_n] = old_emb.weight[:copy_n]
            self.lm_head.weight[:copy_n] = old_head.weight[:copy_n]
            if new_num_tokens > old_n:
                extra = new_num_tokens - old_n
                self.embed_tokens.weight[old_n:] = torch.normal(0, 0.02, (extra, self.config.hidden_size))

        self.lm_head.weight = self.embed_tokens.weight  # re-tie

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        hidden_states = self.embed_tokens(input_ids)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(hidden_states, attention_mask, kv)
            new_kv_caches.append(new_kv)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_kv_caches if kv_caches is not None else None
