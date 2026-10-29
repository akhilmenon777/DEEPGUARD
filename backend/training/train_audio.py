"""
Training script for 2D Mel-Spectrogram CNN Audio Deepfake Classifier.
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

from backend.models.audio_cnn import SpectrogramCNN
from backend.training.dataset import AudioSpoofDataset
from backend.config import AUDIO_CNN_WEIGHTS_PATH

def train_audio_model(
    dataset: AudioSpoofDataset,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_path: str = None,
    device: str = None
):
    save_path = save_path or str(AUDIO_CNN_WEIGHTS_PATH)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=======================================================")
    print(f" Starting Mel-Spectrogram CNN Training on {device.upper()}")
    print(f"=======================================================")

    model = SpectrogramCNN(in_channels=1, num_classes=2)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" SpectrogramCNN Trainable Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

    model.train()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (spectrograms, labels) in enumerate(loader):
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(spectrograms)
            loss = criterion(logits, labels)

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
            print(f"  [Checkpoint] Saved best Audio CNN model to {save_path}")

    return model

if __name__ == "__main__":
    import argparse
    import sys
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(BASE_DIR))

    parser = argparse.ArgumentParser(description="Train 2D Mel-Spectrogram CNN on Real/Spoofed Audio Dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing 'genuine/' and 'spoof/' subfolders")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--save_path", type=str, default=str(AUDIO_CNN_WEIGHTS_PATH), help="Path to save best weights .pth")
    args = parser.parse_args()

    print(f"Loading audio dataset from: {args.data_dir}")
    ds = AudioSpoofDataset(root_dir=args.data_dir)
    print(f"Loaded {len(ds)} samples.")

    if len(ds) == 0:
        print("Error: No audio files found. Ensure your data_dir contains 'genuine/' and 'spoof/' subdirectories with .wav/.mp3 files.")
        sys.exit(1)

    train_audio_model(
        dataset=ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
