# SKILLS.md — VisionLangModel

> Skills for the from-scratch PaliGemma-style VLM.

---

## Skill 1: Run a baseline training

```bash
cd Vision/VisionLangModel
python src/train.py --epochs 1 --batch_size 4 --grad_accum 4
```

Effective batch 16. First epoch takes ~12 hours on P100.

## Skill 2: Diagnose loss stuck at ~130

**Symptom:** loss not decreasing below 130 after 5k steps.

1. **Verify the projector** is actually training — its weights should
   move at least 1e-3 per step. If frozen, set
   `param.requires_grad=True`.
2. **Check the [IMG] token** is in the tokenizer vocab (id reserved).
3. **Reduce** the LR for the projector to 5e-5 (it's the slowest layer
   to converge).
4. **Add a contrastive warm-up** — for the first 1k steps, compute a
   CLIP-style image-text similarity loss in addition to LM loss.

## Skill 3: Add a new vision encoder

To swap from SigLIP-style to EVA-style:

1. Replace `src/visionEncoder.py` with the new class.
2. Keep the output dim (512) consistent with the projector input.
3. Re-init the projector (different input distribution).
4. Re-train from scratch.

**Pitfall:** changing patch size (16 → 14) changes the patch count
(196 → 256). Update the `[IMG]` token count in `multimodalFusion.py`.

## Skill 4: Tune the projector

The projector is the bottleneck of multimodal training:

```python
# src/multimodalFusion.py
class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim=512, decoder_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.Dropout(0.1),
        )
```

- **Hidden dim** 1024 is the default. Larger = slower training, smaller
  = worse alignment.
- **Dropout 0.1** prevents the projector from memorizing single patches.
- **Two Linear layers** (not one) gives the projector non-linear capacity.

## Skill 5: Replace greedy decode with sampling

```python
# src/train.py — inference block
def generate(model, image, tokenizer, max_len=64, temperature=0.7):
    patches = model.encode_image(image)
    prefix = torch.cat([torch.tensor([IMG_TOKEN_ID]), patches.squeeze(0)], dim=0)
    return model.decoder.generate(prefix, max_len, temperature=temperature)
```

Sampling temperature 0.7 produces more diverse captions than greedy.

## Pitfalls
- **BF16 + gradient checkpointing** is mandatory for P100 16 GB. Without
  it the model OOMs at batch 2.
- **`num_workers=2`** for the COCO loader — P100 has limited CPU.
- **COCO captions** are short (~10 tokens). Generation beyond ~30 tokens
  is hallucination.
- **`[IMG]` token** must be a single special token (not multiple) — the
  decoder uses its id to mark the prefix/image boundary.

