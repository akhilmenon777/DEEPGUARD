"""
Quick test script to diagnose video detection issues
"""
import cv2
import numpy as np
from pathlib import Path

# Test video path
video_path = "sample_data/sample_video.mp4"

print("=" * 60)
print("VIDEO DETECTION DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Can we open the video?
print(f"\n1. Testing video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ FAILED: Could not open video")
    exit(1)

print("✓ Video opened successfully")

# Test 2: Video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n2. Video properties:")
print(f"   Total frames: {total_frames}")
print(f"   FPS: {fps}")
print(f"   Resolution: {width}x{height}")

# Test 3: Face detection
print(f"\n3. Testing face detection on frames...")

# Initialize face cascade
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print("❌ FAILED: Haar Cascade not loaded")
    exit(1)

print("✓ Face detector loaded")

# Test first 10 frames
frames_to_test = min(10, total_frames)
faces_found_count = 0

for i in range(frames_to_test):
    ret, frame = cap.read()
    if not ret:
        print(f"   Frame {i}: Could not read")
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try lenient detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        print(f"   Frame {i}: ✓ Found {len(faces)} face(s)")
        faces_found_count += 1
    else:
        print(f"   Frame {i}: ✗ No faces detected")

cap.release()

print(f"\n4. Summary:")
print(f"   Tested: {frames_to_test} frames")
print(f"   Faces found in: {faces_found_count} frames")

if faces_found_count == 0:
    print("\n❌ ISSUE: No faces detected in any frame!")
    print("\nPossible causes:")
    print("   - Face detection cascade not working properly")
    print("   - Video quality too low")
    print("   - Faces not frontal enough")
    print("\nTrying alternate detection method...")
    
    # Try with even more lenient settings
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(10, 10)
        )
        print(f"   Ultra-lenient detection: {len(faces)} faces")
    cap.release()
else:
    print(f"\n✓ SUCCESS: Face detection is working!")

print("\n" + "=" * 60)
