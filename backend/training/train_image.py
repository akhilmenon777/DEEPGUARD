"""
Training and Fine-tuning script for GenD Vision Transformer.
Only LayerNorm parameters and Hyperspherical Head are fine-tuned (~0.03% of weights).
Uses joint Cross-Entropy and Supervised Contrastive Loss.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.models.gend_vit import GenDStyleViT
from backend.training.losses import SupConLoss
from backend.training.dataset import FaceRealFakeDataset
from backend.config import GEND_WEIGHTS_PATH

def train_image_model(
    dataset: FaceRealFakeDataset,
    epochs: int = 5,
    batch_size: int = 8,
    lr_head: float = 1e-3,
    lr_ln: float = 1e-4,
    lambda_supcon: float = 0.5,
    save_path: str = None,
    device: str = None
):
    save_path = save_path or str(GEND_WEIGHTS_PATH)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=======================================================")
    print(f" Starting GenD ViT LayerNorm Fine-Tuning on {device.upper()}")
    print(f"=======================================================")

    model = GenDStyleViT(model_name="vit_tiny_patch16_224", pretrained=False)
    model.to(device)

    # Trainable parameter verification
    param_summary = model.get_trainable_parameter_summary()
    print(f" Total Parameters: {param_summary['total_parameters']:,}")
    print(f" Trainable Parameters: {param_summary['trainable_parameters']:,} ({param_summary['trainable_ratio_pct']}% of weights)")

    # Group parameters: LayerNorms get small lr, head gets standard lr
    ln_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.embed_head.parameters()) + list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": ln_params, "lr": lr_ln, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-2}
    ])

    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

    model.train()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, embeddings = model(images)

            loss_ce = criterion_ce(logits, labels)
            loss_supcon = criterion_supcon(embeddings, labels)
            loss = loss_ce + lambda_supcon * loss_supcon

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / max(1, total)
        acc = (correct / max(1, total)) * 100.0
        print(f" Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "loss": avg_loss
            }, save_path)
            print(f"  [Checkpoint] Saved best GenD model to {save_path}")

    return model

if __name__ == "__main__":
    import argparse
    import sys
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(BASE_DIR))

    parser = argparse.ArgumentParser(description="Train GenD Vision Transformer on Real/Fake Face Dataset (e.g. 140k dataset) or Kaggle DFDC")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset, e.g. 140k faces folder or DFDC folder")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: 'train', 'valid', 'test', or 'all' (default: 'train')")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to train on (default: None = all)")
    parser.add_argument("--precropped", action="store_true", default=True, help="Skip Haar face detection if faces are already pre-cropped (default: True)")
    parser.add_argument("--no_precropped", dest="precropped", action="store_false", help="Enable Haar face detection on uncropped images")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for projection head")
    parser.add_argument("--lr_ln", type=float, default=1e-4, help="Learning rate for LayerNorm parameters")
    parser.add_argument("--frames_per_video", type=int, default=5, help="For DFDC videos: number of face frames to sample per video (default: 5)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache extracted DFDC faces (optional)")
    parser.add_argument("--save_path", type=str, default=str(GEND_WEIGHTS_PATH), help="Path to save best weights .pth")
    args = parser.parse_args()

    split_arg = None if args.split.lower() == "all" else args.split
    print(f"Loading dataset from: {args.data_dir} (split: {args.split}, max_samples: {args.max_samples})")
    ds = FaceRealFakeDataset(
        root_dir=args.data_dir,
        split=split_arg,
        max_samples=args.max_samples,
        is_precropped=args.precropped,
        frames_per_video=args.frames_per_video,
        cache_dir=args.cache_dir
    )
    print(f"Loaded {len(ds)} samples.")

    if len(ds) == 0:
        print("Error: No samples found. Ensure your data_dir contains:")
        print("  Option 1: Kaggle 140k dataset with train/valid/test folders (or 'real/' and 'fake/' folders)")
        print("  Option 2: Kaggle DFDC directory containing 'metadata.json' and .mp4 videos")
        sys.exit(1)

    train_image_model(
        dataset=ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_ln=args.lr_ln,
        save_path=args.save_path
    )
