# AGENTS.md — VisionLangModel

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific rules only.

> **Project:** `Vision/VisionLangModel/` · **Type:** multimodal VLM
> **Architecture:** SigLIP ViT encoder + Gemma decoder + linear projector
> **Task:** image captioning on COCO 2014 val (~40K image-caption pairs)
> **Hardware:** P100 16GB · **Status:** epoch 1 trained
> **Architecture detail:** see `README.md §13`; cross-project guidance at
> `.agents/skills/computer-vision-multimodal/SKILL.md §2`.

## 1. Subagent: `multimodal-vlm-trainer`

**Triggers:** "Vision-language model not aligning", "Image-text projector
not learning", "VLM loss stuck at 130", "How to inject image patches into
a decoder", "Gemma decoder with vision encoder."

**Knows cold:**
- PaliGemma-style VLM from scratch (no pretrained weights, no high-level
  wrappers). Learns to generate captions by jointly training a SigLIP-
  inspired vision encoder and a Gemma-inspired language decoder connected
  by a learned linear projector.
- **SigLIP Vision Encoder:** Conv2D patch embedding (16×16 → 196 tokens),
  sinusoidal positions, 8× VisionEncoderLayer (pre-norm LayerNorm, 8-head
  self-attention, GELU MLP). Output: (B, 196, 512).
- **Multimodal Projector:** Linear 512 → 1024 + Dropout 0.1.
- **Gemma Language Decoder:** token embeddings (vocab 32K, d_model 1024),
  image patches injected at `[IMG]` placeholders, 12× GemmaDecoderLayer
  (RMSNorm, GQA 8 Q / 4 KV / head_dim 128, RoPE, causal mask, GeGLU FFN),
  final RMSNorm → LM head.
- Training: AdamW (lr 1e-4), gradient accumulation (effective batch 16),
  BF16 mixed precision, gradient checkpointing for P100 (16 GB) fit. COCO
  auto-downloads on first run.
- Results (epoch 1): avg loss 129.19 (range 104.1–161.1), ~202,500
  batches/epoch. High absolute loss expected for a from-scratch model
  jointly aligning 196 patches with free-form captions.

## 2. Hard rules

1. **Never** use a pretrained SigLIP or PaliGemma checkpoint — project is
   from-scratch only.
2. **Always** inject image patches via the `[IMG]` placeholder token. Don't
   concatenate patches to the prefix.
3. **Always** freeze the patch embedding layer for the first 1000 steps
   (it destabilizes early).
4. **Always** use GQA in the decoder (matches Gemma-2 design).
5. **Always** verify the projector output dimension matches the decoder
   `d_model` (both 1024 here).

## 3. Cross-references

- For LLM-side architecture (GQA, RoPE): `LLM/LLaMA-3-Lite/AGENTS.md`.
- For training loop patterns: `.agents/skills/pytorch-deep-dive/SKILL.md`.

## 4. Files

- `src/{visionEncoder,languageDecoder,multimodalFusion,train}.py`.
- `assets/loss_curve.png`.
- `results/training_log.md`.
