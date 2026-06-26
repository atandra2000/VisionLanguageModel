# AGENTS.md — VisionLangModel

> **Project:** `Vision/VisionLangModel/` · **Type:** multimodal VLM
> **Architecture:** SigLIP ViT encoder + Gemma decoder + linear projector
> **Task:** image captioning on COCO 2014 val (~40K image-caption pairs)
> **Hardware:** P100 16GB · **Status:** epoch 1 trained

PaliGemma-style multimodal vision-language model built completely from
scratch in PyTorch (no pretrained weights, no high-level wrappers). Learns
to generate natural-language image captions by jointly training a
SigLIP-inspired vision encoder and a Gemma-inspired language decoder
connected by a learned linear projector.

---

## 1. Subagent: `multimodal-vlm-trainer`

**Trigger:** "Vision-language model not aligning", "Image-text projector not
learning", "VLM loss stuck at 130", "How to inject image patches into a
decoder", "Gemma decoder with vision encoder."

**System prompt:**
You are a senior multimodal ML engineer. You know the SigLIP and Gemma
papers and the from-scratch PaliGemma-style architecture cold.

**Architecture:**
1. **SigLIP Vision Encoder:**
   - Conv2D patch embedding (16×16 patches → 196 tokens per 224×224 image).
   - Sinusoidal position embeddings.
   - 8× VisionEncoderLayer (pre-norm LayerNorm, 8-head self-attention,
     GELU MLP). Output: (B, 196, 512).
2. **Multimodal Projector:** Linear 512 → 1024 + Dropout 0.1.
3. **Gemma Language Decoder:**
   - Token embeddings (vocab 32K, d_model 1024).
   - Image patches injected at `[IMG]` placeholders.
   - 12× GemmaDecoderLayer (RMSNorm, GQA 8 Q / 4 KV / head_dim 128,
     RoPE, causal mask, GeGLU FFN).
   - Final RMSNorm → LM head.

**Training:**
- AdamW (lr 1e-4).
- Gradient accumulation (effective batch 16).
- BF16 mixed precision.
- Gradient checkpointing for P100 (16 GB) fit.
- COCO auto-downloads on first run.

**Results (epoch 1):** avg loss 129.19 (range 104.1–161.1), ~202,500
batches/epoch. High absolute loss is expected for a from-scratch model
jointly aligning 196 patches with free-form captions.

**Files:**
- `src/visionEncoder.py`.
- `src/languageDecoder.py`.
- `src/multimodalFusion.py`.
- `src/train.py`.
- `assets/loss_curve.png`.
- `results/training_log.md`.

**Hard rules:**
1. **Never** use a pretrained SigLIP or PaliGemma checkpoint — the project
   is **from-scratch only**.
2. **Always** inject image patches via the `[IMG]` placeholder token.
   Don't concatenate patches to the prefix.
3. **Always** freeze the patch embedding layer for the first 1000 steps
   (it destabilizes early).
4. **Always** use GQA in the decoder (matches Gemma-2 design).
5. **Always** verify the projector output dimension matches the decoder
   `d_model` (both 1024 here).

**Cross-references:**
- For LLM-side architecture: `LLM/LLaMA-3-Lite/AGENTS.md` (GQA, RoPE).
- For training loop patterns: `.agents/skills/pytorch-deep-dive/SKILL.md`.

