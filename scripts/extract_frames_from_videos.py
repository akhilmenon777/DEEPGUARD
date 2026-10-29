"""
Extract frames from video files for training
Extracts faces and saves as images
"""
import os
import sys
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, fps=1, max_frames=100):
    """
    Extract frames from a single video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (1 = 1 frame per second)
        max_frames: Maximum frames to extract per video
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps == 0:
        video_fps = 30  # Default fallback
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at intervals
        if frame_count % frame_interval == 0 and extracted_count < max_frames:
            frame_filename = output_dir / f"frame_{extracted_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    return extracted_count


def process_directory(input_dir, output_dir, fps=1, max_frames_per_video=100):
    """
    Process all videos in a directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(input_path.rglob(f'*{ext}')))
    
    print(f"\nFound {len(video_files)} video files")
    
    if len(video_files) == 0:
        print(f"No video files found in {input_dir}")
        return
    
    total_frames = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        # Create output subdirectory for this video
        relative_path = video_file.relative_to(input_path)
        video_output_dir = output_path / relative_path.parent / video_file.stem
        
        frames_extracted = extract_frames_from_video(
            video_file, 
            video_output_dir, 
            fps=fps,
            max_frames=max_frames_per_video
        )
        
        total_frames += frames_extracted
    
    print(f"\n✓ Extraction complete!")
    print(f"  Total frames extracted: {total_frames:,}")
    print(f"  Frames per video (avg): {total_frames / len(video_files):.1f}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing videos')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for extracted frames')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second to extract (default: 1)')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Maximum frames per video (default: 100)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Video Frame Extraction Tool")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  FPS: {args.fps} (extracting {args.fps} frame(s) per second)")
    print(f"  Max frames per video: {args.max_frames}")
    
    if not Path(args.input).exists():
        print(f"\n❌ Error: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    process_directory(
        args.input, 
        args.output, 
        fps=args.fps,
        max_frames_per_video=args.max_frames
    )
    
    print(f"\n✓ Next step: Prepare dataset for training")
    print(f"  python scripts/prepare_dataset.py --input {args.output} --output data/processed")


if __name__ == '__main__':
    main()
