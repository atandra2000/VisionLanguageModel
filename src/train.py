"""
train.py — Training script for the VisionLangModel (PaliGemma-style).

Usage:
    python src/train.py
    python src/train.py --epochs 10 --lr 3e-4 --accum-steps 8
"""

import argparse
import os
import json
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import requests

# ── Allow memory-efficient CUDA allocation ────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.insert(0, os.path.dirname(__file__))

from visionEncoder import VisionConfig
from languageDecoder import LanguageConfig
from multimodalFusion import create_optimized_paligemma, optimize_for_p100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_coco_dataset(dataset_dir: str = "coco_dataset"):
    os.makedirs(dataset_dir, exist_ok=True)

    images_url = "http://images.cocodataset.org/zips/val2014.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

    def _fetch_and_extract(url, zip_path, extract_dir):
        if not os.path.exists(zip_path):
            print(f"Downloading {os.path.basename(zip_path)}...")
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        if not os.path.exists(extract_dir):
            print(f"Extracting {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dataset_dir)

    _fetch_and_extract(
        images_url,
        os.path.join(dataset_dir, "val2014.zip"),
        os.path.join(dataset_dir, "val2014"),
    )
    _fetch_and_extract(
        annotations_url,
        os.path.join(dataset_dir, "annotations_trainval2014.zip"),
        os.path.join(dataset_dir, "annotations"),
    )
    return (
        os.path.join(dataset_dir, "val2014"),
        os.path.join(dataset_dir, "annotations", "captions_val2014.json"),
    )


def load_coco_annotations(annotation_file: str):
    with open(annotation_file) as f:
        data = json.load(f)
    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    return [
        {"image": id_to_fname[ann["image_id"]], "caption": ann["caption"]}
        for ann in data["annotations"]
    ]


class MultimodalDataset(Dataset):
    def __init__(self, image_dir: str, annotations: list, vision_cfg: VisionConfig, lang_cfg: LanguageConfig, tokenizer):
        self.annotations = annotations
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        num_patches = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
        self.max_text_len = lang_cfg.max_position_embeddings - num_patches - 2
        self.image_token_id = tokenizer.convert_tokens_to_ids("[IMG]")
        self.num_patches = num_patches

        self.transform = transforms.Compose([
            transforms.Resize((vision_cfg.image_size, vision_cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        pixel_values = self.transform(
            Image.open(os.path.join(self.image_dir, ann["image"])).convert("RGB")
        )
        text_tokens = self.tokenizer.encode(ann["caption"], add_special_tokens=False)[: self.max_text_len]
        img_tokens = [self.image_token_id] * self.num_patches
        ids = [self.tokenizer.bos_token_id] + img_tokens + text_tokens + [self.tokenizer.eos_token_id]
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, masks = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    torch.cuda.empty_cache()

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer.add_special_tokens({
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "additional_special_tokens": ["[IMG]"],
    })

    vision_cfg = VisionConfig()
    lang_cfg = LanguageConfig(vocab_size=len(tokenizer))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = create_optimized_paligemma(vision_cfg, lang_cfg, tokenizer, device=DEVICE)
    model.language_model.resize_token_embeddings(len(tokenizer))
    model = optimize_for_p100(model, enable_checkpointing=True)
    model.to(DEVICE, dtype=torch.float16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── Data ──────────────────────────────────────────────────────────────────
    image_dir, ann_file = download_coco_dataset()
    annotations = load_coco_annotations(ann_file)
    dataset = MultimodalDataset(image_dir, annotations, vision_cfg, lang_cfg, tokenizer)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)

    # ── Loop ──────────────────────────────────────────────────────────────────
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss, optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )["loss"] / args.accum_steps

            loss.backward()
            total_loss += loss.item() * args.accum_steps

            if (batch_idx + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 500 == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs}  "
                    f"Batch {batch_idx}  "
                    f"Loss: {loss.item() * args.accum_steps:.4f}"
                )

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} completed  Average Loss: {avg:.4f}")

    # ── Sample inference ──────────────────────────────────────────────────────
    print("\nRunning sample inference...")
    model.eval()
    test_img = dataset.transform(
        Image.open(os.path.join(image_dir, annotations[0]["image"])).convert("RGB")
    ).unsqueeze(0).to(DEVICE)
    img_tok_id = tokenizer.convert_tokens_to_ids("[IMG]")
    prompt = torch.tensor(
        [[tokenizer.bos_token_id] + [img_tok_id] * dataset.num_patches],
        dtype=torch.long,
        device=DEVICE,
    )
    with torch.no_grad():
        gen = model.generate(prompt, test_img, max_new_tokens=50, temperature=1.0)
    print("Generated caption:", tokenizer.decode(gen[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum-steps", type=int, default=16)
    train(parser.parse_args())
