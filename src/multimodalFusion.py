from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from visionEncoder import SigLIPVisionEncoder, VisionConfig, VisionEncoderLayer
from languageDecoder import GemmaDecoderLayer, GemmaLanguageModel, LanguageConfig


@dataclass
class PaliGemmaConfig:
    vision_config: VisionConfig
    language_config: LanguageConfig
    vision_output_dim: int = 512
    language_input_dim: int = 1024
    projection_dropout: float = 0.1


class MultimodalProjector(nn.Module):
    """Linear projection bridging the vision encoder output to the language decoder input space."""

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_output_dim, config.language_input_dim)
        self.dropout = nn.Dropout(config.projection_dropout)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(vision_features))


class PaliGemmaModel(nn.Module):
    """
    PaliGemma-style multimodal model.

    Architecture:
      SigLIP Vision Encoder → Linear Projector → Gemma Language Decoder

    Image patches are projected to the language embedding space and
    concatenated with text token embeddings at the [IMG] placeholder positions
    before being passed through the causal decoder.
    """

    def __init__(self, vision_config: VisionConfig, language_config: LanguageConfig, tokenizer=None):
        super().__init__()
        self.config = PaliGemmaConfig(vision_config, language_config)
        self.vision_encoder = SigLIPVisionEncoder(vision_config)
        self.language_model = GemmaLanguageModel(language_config)
        self.multimodal_projector = MultimodalProjector(self.config)

        if tokenizer is not None:
            self.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
            self.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
            self.image_token_id = tokenizer.convert_tokens_to_ids("[IMG]")
        else:
            self.bos_token_id = 0
            self.eos_token_id = 1
            self.image_token_id = language_config.vocab_size - 1

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.multimodal_projector(self.vision_encoder(pixel_values))

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replace [IMG] token positions with projected image patch embeddings."""
        B, T = input_ids.size()
        text_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is None:
            return text_embeds

        image_features = self.encode_images(pixel_values)   # (B, n_patches, d)
        n_patches = image_features.size(1)
        combined = text_embeds.clone()
        img_mask = input_ids == self.image_token_id

        for b in range(B):
            positions = torch.where(img_mask[b])[0]
            if len(positions) == 0:
                continue
            start = positions[0].item()
            end = min(start + n_patches, T)
            combined[b, start:end] = image_features[b, : end - start]

        return combined

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> dict:
        inputs_embeds = self.prepare_inputs_for_multimodal(input_ids, pixel_values)

        # Run through decoder layers manually so we can thread KV caches
        hidden_states = inputs_embeds
        new_kv_caches = []
        for i, layer in enumerate(self.language_model.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(hidden_states, attention_mask, kv)
            new_kv_caches.append(new_kv)

        hidden_states = self.language_model.norm(hidden_states)
        logits = self.language_model.lm_head(hidden_states)

        outputs = {"logits": logits, "kv_caches": new_kv_caches if kv_caches is not None else None}

        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
            )
            outputs["loss"] = loss

        return outputs

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation with optional image conditioning."""
        self.eval()
        current_ids = input_ids.clone()
        kv_caches = [None] * self.config.language_config.num_hidden_layers

        for _ in range(max_new_tokens):
            outputs = self.forward(current_ids, pixel_values, kv_caches=kv_caches)
            logits = outputs["logits"]
            kv_caches = outputs["kv_caches"]

            next_logits = logits[:, -1, :] / temperature
            if do_sample:
                next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if next_token.item() == self.eos_token_id:
                break

        return current_ids


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_optimized_paligemma(
    vision_config: VisionConfig,
    language_config: LanguageConfig,
    tokenizer,
    device: str = "cuda",
) -> PaliGemmaModel:
    model = PaliGemmaModel(vision_config=vision_config, language_config=language_config, tokenizer=tokenizer)
    return model.to(device, dtype=torch.float16)


def optimize_for_p100(model: PaliGemmaModel, enable_checkpointing: bool = True) -> PaliGemmaModel:
    """Apply P100-specific memory optimisations via gradient checkpointing."""
    if not enable_checkpointing:
        return model

    def _checkpointed_forward(module):
        original = module.forward

        def forward(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(original, *args, use_reentrant=True, **kwargs)

        return forward

    for module in model.modules():
        if isinstance(module, (VisionEncoderLayer, GemmaDecoderLayer)):
            module.forward = _checkpointed_forward(module)

    return model
