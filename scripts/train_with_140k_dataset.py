"""
Automated Training and Evaluation Pipeline for GenD Vision Transformer
Trained on Kaggle 140k Real and Fake Faces Dataset (xhlulu/140k-real-and-fake-faces)
Project Authors: Yash Jain (K019), Akhil Menon (K036), Taaksh Patel (K042)
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

from backend.models.gend_vit import GenDStyleViT
from backend.training.losses import SupConLoss
from backend.training.dataset import FaceRealFakeDataset
from backend.detectors.image_video_detector import ImageVideoDetector
from backend.config import GEND_WEIGHTS_PATH

def download_140k_dataset(dest_dir: str, kaggle_user: str = "taakshpatel", kaggle_key: str = "c87903a36495c3edeaae3c9cffccc385"):
    """Downloads and unzips xhlulu/140k-real-and-fake-faces via Kaggle API."""
    os.makedirs(dest_dir, exist_ok=True)
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key

    print("=" * 70)
    print(" [1/3] Downloading Kaggle Dataset: xhlulu/140k-real-and-fake-faces")
    print(f" Target Directory: {dest_dir}")
    print("=" * 70)

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", "xhlulu/140k-real-and-fake-faces",
        "--unzip",
        "-p", dest_dir
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("[Warning] Kaggle CLI encountered an issue. Checking if files are already extracted...")
    else:
        print("Dataset download and extraction complete!")

def train_and_eval_140k(
    data_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    max_train_samples: int = 5000,
    max_test_samples: int = 500,
    save_path: str = None
):
    save_path = save_path or str(GEND_WEIGHTS_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print(" [2/3] GenD Vision Transformer LayerNorm Fine-Tuning")
    print(f" Device: {device.upper()} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f" Max Samples: {max_train_samples or 'All (100k)'} | Weights Destination: {save_path}")
    print("=" * 70)

    # 1. Load Training Dataset
    train_ds = FaceRealFakeDataset(
        root_dir=data_dir,
        split="train",
        max_samples=max_train_samples,
        is_precropped=True
    )
    if len(train_ds) == 0:
        print(f"[Error] No images found in {data_dir}. Please check your path.")
        return False

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=(len(train_ds) > batch_size)
    )

    # 2. Initialize Model
    model = GenDStyleViT(model_name="vit_tiny_patch16_224", pretrained=False)
    model.to(device)

    param_summary = model.get_trainable_parameter_summary()
    print(f" Total Parameters:     {param_summary['total_parameters']:,}")
    print(f" Trainable Parameters: {param_summary['trainable_parameters']:,} ({param_summary['trainable_ratio_pct']}% LayerNorm adaptation)")

    # 3. LayerNorm-only optimization
    ln_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.embed_head.parameters()) + list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": ln_params, "lr": 1e-4, "weight_decay": 1e-4},
        {"params": head_params, "lr": 1e-3, "weight_decay": 1e-2}
    ])
    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07)

    # 4. Training Loop
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, embeddings = model(images)

            loss = criterion_ce(logits, labels) + 0.5 * criterion_supcon(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = (correct / total) * 100.0
        avg_loss = total_loss / total
        print(f" Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} - Train Accuracy: {train_acc:.2f}%")

        if train_acc > best_acc:
            best_acc = train_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"   --> Saved new best checkpoint to {save_path} (Acc: {train_acc:.2f}%)")

    # 5. Evaluate on Test Set
    print("\n" + "=" * 70)
    print(" [3/3] Evaluating GenD on Unseen Test Split")
    print("=" * 70)
    test_ds = FaceRealFakeDataset(
        root_dir=data_dir,
        split="test",
        max_samples=max_test_samples,
        is_precropped=True
    )
    if len(test_ds) > 0:
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                preds = torch.argmax(logits, dim=1)
                t_correct += (preds == labels).sum().item()
                t_total += len(labels)
        test_acc = (t_correct / t_total) * 100.0
        print(f" Test Set Accuracy: {test_acc:.2f}% on {t_total} samples")
    else:
        print("[Note] Test split not found or empty; skipping test evaluation.")

    print("\n" + "=" * 70)
    print(" Training Complete! Model weights are active in backend/weights/gend_vit_best.pth")
    print(" Launch the web application via: py -3.12 scripts/run_server.py")
    print("=" * 70)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated 140k Real/Fake Face Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/140k_faces", help="Path to 140k dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle before training")
    parser.add_argument("--kaggle_user", type=str, default="taakshpatel", help="Kaggle Username")
    parser.add_argument("--kaggle_key", type=str, default="c87903a36495c3edeaae3c9cffccc385", help="Kaggle API Key")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max train samples (default: 5000, 0 = all 100k)")
    parser.add_argument("--save_path", type=str, default=str(GEND_WEIGHTS_PATH), help="Destination weights path")
    args = parser.parse_args()

    max_s = None if args.max_samples <= 0 else args.max_samples

    if args.download:
        download_140k_dataset(args.data_dir, kaggle_user=args.kaggle_user, kaggle_key=args.kaggle_key)

    train_and_eval_140k(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=max_s,
        save_path=args.save_path
    )
