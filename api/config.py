"""
Configuration for Vercel deployment.
"""
import os
from pathlib import Path

# Base paths - adjusted for Vercel environment
BASE_DIR = Path(__file__).resolve().parent.parent
API_DIR = BASE_DIR / "api"
WEIGHTS_DIR = BASE_DIR / "weights"
TEMP_DIR = Path("/tmp")  # Vercel's writable directory

# Image / Video settings
IMG_SIZE = 224
MAX_FRAMES = 20
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Decision Engine Thresholds
VISUAL_AUTHENTIC_THRESHOLD = 0.35
VISUAL_SUSPICIOUS_THRESHOLD = 0.65
AUDIO_SYNTHETIC_THRESHOLD = 0.50

# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 4.0
AUDIO_NUM_MELS = 128
AUDIO_N_FFT = 1024
AUDIO_HOP_LENGTH = 512

# Model weights paths
GEND_WEIGHTS_PATH = WEIGHTS_DIR / "gend_vit_best.pth"
AUDIO_CNN_WEIGHTS_PATH = WEIGHTS_DIR / "audio_cnn_best.pth"
