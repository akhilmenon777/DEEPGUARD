"""
PyTorch Datasets for Visual and Auditory Deepfake Detection.
Supports:
1. Standard image/audio folder structure (real/fake, genuine/spoof)
2. Kaggle Deepfake Detection Challenge (DFDC) dataset (metadata.json + .mp4 videos)
   Reference: https://www.kaggle.com/code/robikscube/kaggle-deepfake-detection-introduction
"""
import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any

from backend.utils.face_utils import extract_face
from backend.utils.audio_utils import load_audio_resampled, compute_mel_spectrogram

class FaceRealFakeDataset(Dataset):
    """
    Unified Dataset for Real vs Fake Face Images.
    Supports:
    1. Kaggle 140k Real and Fake Faces dataset (xhlulu/140k-real-and-fake-faces)
       with train/valid/test splits and real/fake classes.
    2. Folder with direct 'real/' and 'fake/' subdirectories.
    3. Kaggle DFDC directory with 'metadata.json' and '.mp4' videos.
    Label 0: Authentic / Real
    Label 1: Manipulated / Deepfake
    """
    def __init__(
        self,
        samples: Optional[List[Tuple[str, int]]] = None,
        root_dir: Optional[str] = None,
        img_size: int = 224,
        frames_per_video: int = 5,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        is_precropped: bool = True,
        transform=None
    ):
        self.img_size = img_size
        self.transform = transform
        self.is_precropped = is_precropped
        self.samples: List[Tuple[str, int]] = []

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if samples:
            self.samples.extend(samples)

        if root_dir and os.path.exists(root_dir):
            metadata_file = os.path.join(root_dir, "metadata.json")

            # Check if this is a Kaggle DFDC dataset directory
            if os.path.exists(metadata_file):
                print(f"[FaceRealFakeDataset] Detected Kaggle DFDC format at {root_dir}")
                dfdc_samples = self._load_dfdc(root_dir, metadata_file, frames_per_video, cache_dir)
                self.samples.extend(dfdc_samples)
            else:
                self._load_image_folders(root_dir, split)

        if max_samples and max_samples > 0 and len(self.samples) > max_samples:
            import random
            rng = random.Random(42)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"[FaceRealFakeDataset] Subsampled to {len(self.samples)} items.")

    def _load_image_folders(self, root_dir: str, split: Optional[str] = None):
        """Discovers real and fake subfolders, handling 140k dataset nested structures."""
        import glob
        
        target_dirs = []
        if split:
            # Look for e.g. root_dir/train, or root_dir/**/train
            direct_split = os.path.join(root_dir, split)
            if os.path.exists(direct_split):
                target_dirs.append(direct_split)
            else:
                nested = glob.glob(os.path.join(root_dir, "**", split), recursive=True)
                for d in nested:
                    if os.path.isdir(d):
                        target_dirs.append(d)

        if not target_dirs:
            # Fall back to root_dir directly
            target_dirs.append(root_dir)

        real_count = 0
        fake_count = 0

        for base in target_dirs:
            # Check for direct real and fake folders
            real_dir = os.path.join(base, "real")
            fake_dir = os.path.join(base, "fake")

            # If not direct, search recursively within base
            if not os.path.exists(real_dir):
                found_reals = glob.glob(os.path.join(base, "**", "real"), recursive=True)
                real_dir = found_reals[0] if len(found_reals) > 0 else None
            if not os.path.exists(fake_dir):
                found_fakes = glob.glob(os.path.join(base, "**", "fake"), recursive=True)
                fake_dir = found_fakes[0] if len(found_fakes) > 0 else None

            if real_dir and os.path.exists(real_dir):
                for fname in os.listdir(real_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        self.samples.append((os.path.join(real_dir, fname), 0))
                        real_count += 1

            if fake_dir and os.path.exists(fake_dir):
                for fname in os.listdir(fake_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        self.samples.append((os.path.join(fake_dir, fname), 1))
                        fake_count += 1

        print(f"[FaceRealFakeDataset] Loaded {real_count} real and {fake_count} fake images (total: {len(self.samples)}).")

    def _load_dfdc(self, root_dir: str, metadata_file: str, frames_per_video: int, cache_dir: Optional[str]) -> List[Tuple[str, int]]:
        """
        Parses Kaggle DFDC metadata.json and extracts face crops from .mp4 videos.
        Caches extracted face crops to avoid re-extracting on every epoch.
        """
        cache_folder = cache_dir or os.path.join(root_dir, "dfdc_face_cache")
        os.makedirs(cache_folder, exist_ok=True)

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        extracted_samples = []
        video_files = [v for v in metadata.keys() if os.path.exists(os.path.join(root_dir, v))]

        print(f"[DFDC Loader] Parsing {len(video_files)} videos from metadata.json...")

        for vname in video_files:
            meta = metadata[vname]
            label_str = meta.get("label", "FAKE").upper()
            label = 0 if label_str == "REAL" else 1

            vpath = os.path.join(root_dir, vname)
            vstem = Path(vname).stem

            # Check if faces are already cached
            cached_faces = [
                os.path.join(cache_folder, f)
                for f in os.listdir(cache_folder)
                if f.startswith(f"{vstem}_f") and f.endswith(".jpg")
            ]

            if len(cached_faces) >= frames_per_video:
                for cf in cached_faces[:frames_per_video]:
                    extracted_samples.append((cf, label))
                continue

            # Extract faces from video frames
            cap = cv2.VideoCapture(vpath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                continue

            interval = max(1, total_frames // (frames_per_video * 2))
            saved_count = 0
            frame_idx = 0

            while cap.isOpened() and saved_count < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    face_rgb, _, face_found = extract_face(frame, target_size=self.img_size)
                    if face_found or saved_count == 0:
                        save_name = f"{vstem}_f{saved_count}_l{label}.jpg"
                        save_path = os.path.join(cache_folder, save_name)
                        cv2.imwrite(save_path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
                        extracted_samples.append((save_path, label))
                        saved_count += 1

                frame_idx += 1

            cap.release()

        print(f"[DFDC Loader] Extracted/Loaded {len(extracted_samples)} face samples.")
        return extracted_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Extract face if not already pre-cropped
        if self.is_precropped or "_f" in os.path.basename(img_path):
            face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if face_rgb.shape[:2] != (self.img_size, self.img_size):
                face_rgb = cv2.resize(face_rgb, (self.img_size, self.img_size))
        else:
            face_rgb, _, _ = extract_face(img_bgr, target_size=self.img_size)

        norm = face_rgb.astype(np.float32) / 255.0
        norm = (norm - self.mean) / self.std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).float()

        return tensor, torch.tensor(label, dtype=torch.long)


class AudioSpoofDataset(Dataset):
    """
    Dataset for Speech Deepfakes and Spoof Detection.
    Supports:
    1. Folder with 'genuine/' and 'spoof/' subdirectories.
    2. Direct audio clips or audio extracted from DFDC videos.
    Label 0: Genuine Speech
    Label 1: Spoofed / Cloned / Synthetic Audio
    """
    def __init__(self, samples: Optional[List[Tuple[str, int]]] = None, root_dir: Optional[str] = None, sample_rate: int = 16000, duration: float = 4.0, n_mels: int = 128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.samples: List[Tuple[str, int]] = []

        if samples:
            self.samples.extend(samples)

        if root_dir and os.path.exists(root_dir):
            genuine_dir = os.path.join(root_dir, "genuine")
            spoof_dir = os.path.join(root_dir, "spoof")

            if os.path.exists(genuine_dir):
                for fname in os.listdir(genuine_dir):
                    if fname.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                        self.samples.append((os.path.join(genuine_dir, fname), 0))

            if os.path.exists(spoof_dir):
                for fname in os.listdir(spoof_dir):
                    if fname.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                        self.samples.append((os.path.join(spoof_dir, fname), 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        audio_path, label = self.samples[idx]
        sig, sr = load_audio_resampled(audio_path, target_sr=self.sample_rate, duration_sec=self.duration)
        mel_norm, _ = compute_mel_spectrogram(sig, sr=sr, n_mels=self.n_mels)
        tensor = torch.from_numpy(mel_norm).unsqueeze(0).float() # (1, 128, T)
        return tensor, torch.tensor(label, dtype=torch.long)
