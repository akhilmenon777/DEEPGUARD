"""
Audio Deepfake Spoof Classifier using 2D Mel-Spectrogram CNN.
Matches Project Design: 3 Conv blocks + Linear classifier.
"""
import torch
import torch.nn as nn
from backend.utils.audio_utils import load_audio_resampled, compute_mel_spectrogram

class SpectrogramCNN(nn.Module):
    """
    3-Block Convolutional Neural Network operating on 128-band Log Mel-Spectrograms.
    Detects acoustic artifacts, phase incoherence, and high-frequency vocoder distortions.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        # Three convolution blocks as per specification
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 1, 128, T)
        Returns:
            logits: Tensor of shape (B, 2)
        """
        feats = self.conv(x)
        pooled = self.pool(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits


def melspectrogram_from_file(file_path: str, sample_rate: int = 16000, duration: float = 4.0, n_mels: int = 128):
    """
    Reads audio from file, resamples to 16kHz, pads/trims to 4.0s,
    and returns a normalized 128-band log-mel spectrogram tensor and raw mel data.
    """
    signal_data, sr = load_audio_resampled(file_path, target_sr=sample_rate, duration_sec=duration)
    mel_norm, log_mel = compute_mel_spectrogram(signal_data, sr=sr, n_mels=n_mels)
    tensor = torch.from_numpy(mel_norm).unsqueeze(0).unsqueeze(0) # (1, 1, 128, T)
    return tensor, log_mel
