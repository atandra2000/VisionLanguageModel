# Training Log — COCO 2014 Validation Set

**Hardware:** NVIDIA Tesla P100 (16 GB)
**Platform:** Kaggle Notebooks
**Dataset:** COCO 2014 Validation (~40k image–caption pairs)
**Batch size:** 1 · Gradient accumulation: 16 steps (effective batch ≈ 16)
**Optimiser:** AdamW · LR: 1e-4
**Mixed precision:** `bfloat16` via `torch.amp.autocast`

---

## Epoch Summary

| Epoch | Avg Loss | Notes |
|:-----:|:--------:|:------|
| 1/20  | 129.19   | Initial learning — model acquires coarse image–text alignment |
| 2/20  | —        | Run truncated (Kaggle session time limit) |

---

## Epoch 1 Batch-Level Highlights

| Batch   | Loss    |
|--------:|--------:|
| 0       | 139.88  |
| 500     | 108.03  |
| 10,000  | 121.34  |
| 50,000  | 124.43  |
| 100,000 | 118.46  |
| 150,000 | 112.79  |
| 200,000 | 122.50  |
| 202,500 | 127.87  |

> Loss fluctuates due to high variance across image–caption pairs in COCO.
> The running average trends from ~135 at the start of the epoch to ~125 by the end.

---

## Notes

- The model was not pre-trained; weights were randomly initialised.
- The large loss magnitude reflects that the model is learning to jointly align
  196 image patches (14×14 grid from 224px images with patch_size=16) with
  free-form natural language captions from scratch.
- Gradient checkpointing (`torch.utils.checkpoint`) was enabled to fit the
  combined vision + language model on a single P100.
