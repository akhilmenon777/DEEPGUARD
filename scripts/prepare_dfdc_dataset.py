"""
Kaggle DFDC Dataset Preprocessor.
Extracts faces and audio tracks from the Deepfake Detection Challenge dataset
(as used in Rob Mulla's Kaggle notebook: https://www.kaggle.com/code/robikscube/kaggle-deepfake-detection-introduction)
into organized training folders for GenD ViT and Mel-Spectrogram CNN.
"""
import os
import sys
import json
import cv2
import argparse
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.utils.face_utils import extract_face

def process_dfdc_dataset(
    dfdc_dir: str,
    output_images_dir: str = "data_dfdc_faces",
    output_audio_dir: str = "data_dfdc_audio",
    frames_per_video: int = 5,
    max_videos: int = None
):
    dfdc_path = Path(dfdc_dir)
    meta_path = dfdc_path / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dfdc_dir}. Ensure you point to the DFDC video directory.")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print("=" * 65)
    print(f" Processing Kaggle DFDC Dataset from: {dfdc_dir}")
    print(f" Total entries in metadata.json: {len(metadata)}")
    print("=" * 65)

    real_img_out = Path(output_images_dir) / "real"
    fake_img_out = Path(output_images_dir) / "fake"
    real_img_out.mkdir(parents=True, exist_ok=True)
    fake_img_out.mkdir(parents=True, exist_ok=True)

    videos = [v for v in metadata.keys() if (dfdc_path / v).exists()]
    if max_videos:
        videos = videos[:max_videos]

    print(f"Found {len(videos)} videos on disk to process.")

    real_count = 0
    fake_count = 0
    total_faces = 0

    for i, vname in enumerate(videos, 1):
        meta = metadata[vname]
        label = meta.get("label", "FAKE").upper()
        target_dir = real_img_out if label == "REAL" else fake_img_out

        vpath = str(dfdc_path / vname)
        vstem = Path(vname).stem

        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        step = max(1, total_frames // (frames_per_video * 2))
        saved = 0
        f_idx = 0

        while cap.isOpened() and saved < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            if f_idx % step == 0:
                face_rgb, _, found = extract_face(frame, target_size=224)
                if found or saved == 0:
                    out_name = f"{vstem}_f{saved}.jpg"
                    out_path = target_dir / out_name
                    cv2.imwrite(str(out_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
                    saved += 1
                    total_faces += 1

            f_idx += 1

        cap.release()

        if label == "REAL":
            real_count += 1
        else:
            fake_count += 1

        if i % 10 == 0 or i == len(videos):
            print(f"[{i}/{len(videos)}] Processed {vname} ({label}) -> Total faces extracted: {total_faces}")

    print("\n" + "=" * 65)
    print(" DFDC Extraction Complete!")
    print(f" Real Videos: {real_count} | Fake Videos: {fake_count}")
    print(f" Faces saved to:")
    print(f"   Real faces: {real_img_out} ({len(list(real_img_out.glob('*.jpg')))} images)")
    print(f"   Fake faces: {fake_img_out} ({len(list(fake_img_out.glob('*.jpg')))} images)")
    print("=" * 65)
    print(f"You can now train GenD ViT with:")
    print(f"  py -3.12 backend/training/train_image.py --data_dir {output_images_dir} --epochs 5")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from Kaggle DFDC videos for training")
    parser.add_argument("--dfdc_dir", type=str, required=True, help="Path to Kaggle DFDC directory containing metadata.json and .mp4 files")
    parser.add_argument("--output_dir", type=str, default="data_dfdc_faces", help="Output directory to save real/ and fake/ faces")
    parser.add_argument("--frames_per_video", type=int, default=5, help="Number of face frames to sample per video")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process (optional)")
    args = parser.parse_args()

    process_dfdc_dataset(
        dfdc_dir=args.dfdc_dir,
        output_images_dir=args.output_dir,
        frames_per_video=args.frames_per_video,
        max_videos=args.max_videos
    )
