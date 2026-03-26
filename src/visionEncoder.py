import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 512          # Reduced from 768
    num_hidden_layers: int = 8      # Reduced from 12
    num_attention_heads: int = 8    # Reduced from 12
    intermediate_size: int = 1536   # Reduced from 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6


class PatchEmbedding(nn.Module):
    """Converts image patches into embedding vectors following ViT methodology."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2
        # Convolutional layer acts as patch extraction and linear projection
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Input:  (B, C, H, W)
        batch_size = pixel_values.size(0)
        # (B, hidden_size, n_h, n_w) → flatten → (B, hidden_size, n_patches)
        embeddings = self.projection(pixel_values).flatten(2)
        # (B, n_patches, hidden_size)
        return embeddings.transpose(1, 2)


class VisionAttention(nn.Module):
    """Multi-head self-attention mechanism optimised for vision tasks."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores += attention_mask[:, None, None, :].to(scores.device)

        attn_probs = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return self.output_projection(context)


class VisionMLP(nn.Module):
    """Feed-forward network with GELU activation for vision transformer."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.dropout(F.gelu(self.dense1(hidden_states))))


class VisionEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm architecture."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attention = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        hidden_states = hidden_states + self.attention(self.layer_norm1(hidden_states), attention_mask)
        # Pre-norm FFN
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP-style vision encoder: patch embedding → positional encoding
    → stacked transformer layers → final LayerNorm.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embedding.num_patches, config.hidden_size)
        )
        self.encoder_layers = nn.ModuleList(
            [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_position_embeddings()

    def _init_position_embeddings(self):
        n = self.patch_embedding.num_patches
        d = self.config.hidden_size
        position = torch.arange(n).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * -(math.log(10000.0) / d)
        )
        pos_emb = torch.zeros(n, d)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.position_embeddings.data = pos_emb.unsqueeze(0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.patch_embedding(pixel_values) + self.position_embeddings)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.layer_norm(x)
