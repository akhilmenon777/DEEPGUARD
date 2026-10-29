"""
Unit tests for Audio Spoof Classifier and Mel-Spectrogram pipeline.
"""
import torch
import numpy as np
from backend.models.audio_cnn import SpectrogramCNN
from backend.utils.audio_utils import compute_mel_spectrogram, analyze_spectral_heuristics

def test_spectrogram_cnn_forward():
    model = SpectrogramCNN(in_channels=1, num_classes=2)
    model.eval()

    # Input batch: (B=2, C=1, Mels=128, Time=126)
    dummy_mel = torch.randn(2, 1, 128, 126)
    logits = model(dummy_mel)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()

def test_mel_spectrogram_extraction():
    sr = 16000
    duration = 4.0
    signal_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))

    mel_norm, log_mel = compute_mel_spectrogram(signal_data, sr=sr, n_mels=128)

    assert mel_norm.shape[0] == 128
    assert mel_norm.shape[1] > 0
    # Mean of z-score normalized mel should be close to 0
    assert abs(np.mean(mel_norm)) < 1e-4

def test_spectral_heuristics():
    dummy_log_mel = np.random.randn(128, 126) * 10 - 20
    notes, advice = analyze_spectral_heuristics(dummy_log_mel)

    assert isinstance(notes, str) and len(notes) > 0
    assert isinstance(advice, str) and len(advice) > 0
