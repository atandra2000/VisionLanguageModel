"""Smoke test for VisionLangModel (PaliGemma-style architecture)."""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionEncoder import VisionConfig
from languageDecoder import LanguageConfig
from multimodalFusion import PaliGemmaModel


def test_vision_lang_model_smoke():
    # Miniature configuration for CPU smoke test
    v_cfg = VisionConfig(
        image_size=64,
        patch_size=16,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )
    l_cfg = LanguageConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
    )

    model = PaliGemmaModel(v_cfg, l_cfg)
    model.eval()

    batch_size = 2
    # image_size=64, patch_size=16 -> (64/16)^2 = 16 patches
    pixel_values = torch.randn(batch_size, 3, 64, 64)
    # text input: 16 [IMG] tokens + 4 text tokens = 20 tokens
    img_tok = model.image_token_id
    input_ids = torch.cat([
        torch.full((batch_size, 16), img_tok, dtype=torch.long),
        torch.randint(10, 500, (batch_size, 4), dtype=torch.long),
    ], dim=1)

    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, 20, l_cfg.vocab_size)
