# Download Real Deepfake Datasets - Step by Step Guide

## Recommended: FaceForensics++ Dataset

This is the industry-standard dataset used in research papers.

### Step 1: Request Access (5 minutes)

1. **Visit the official GitHub**: https://github.com/ondyari/FaceForensics
2. **Fill out the form**: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
3. **Wait for email** (usually arrives within 24-48 hours)
4. **You'll receive download links** via email

### Step 2: Download Dataset (1-3 hours depending on internet)

Once you receive the email with download links:

#### Option A: Download via Script (Easiest)

The email will contain a Python script. Use it:

```bash
# They'll provide a script like this:
python download-FaceForensics.py \
    -d FaceForensics++ \
    -c c23 \
    -t images \
    --output_path C:/Users/user/Downloads/deepfake-detection-system/deepfake-detection-system/data/faceforensics_raw
```

#### Option B: Direct Download Links

They may also provide direct download links. Download these files:
- `real.zip` (original videos/images)
- `Deepfakes.zip` (face-swapped videos)
- `Face2Face.zip` (reenactment videos)
- `FaceSwap.zip` (another face-swap method)
- `NeuralTextures.zip` (texture-based manipulation)

Download location: `C:/Users/user/Downloads/deepfake-detection-system/deepfake-detection-system/data/faceforensics_raw`

### Step 3: Extract and Organize (30 minutes)

After downloading, extract all zip files:

```bash
# Windows users: Right-click each zip → Extract All
# Or use PowerShell:
cd data\faceforensics_raw
Expand-Archive real.zip -DestinationPath real
Expand-Archive Deepfakes.zip -DestinationPath fake
```

Expected structure after extraction:
```
data/faceforensics_raw/
├── real/
│   ├── video1/
│   ├── video2/
│   └── ...
└── fake/
    ├── video1/
    ├── video2/
    └── ...
```

### Step 4: Extract Frames from Videos (1-2 hours)

If you downloaded videos, extract frames:

```bash
python scripts/extract_frames_from_videos.py \
    --input data/faceforensics_raw \
    --output data/faceforensics_frames \
    --fps 1
```

This extracts 1 frame per second from each video.

### Step 5: Prepare for Training (30 minutes)

```bash
python scripts/prepare_dataset.py \
    --input data/faceforensics_frames \
    --output data/processed \
    --real-dir real \
    --fake-dir fake \
    --resize \
    --target-size 224
```

This will:
- Resize all images to 224x224
- Detect and crop faces
- Split into train/val/test (70/15/15)
- Organize into proper structure

Expected output:
```
Dataset Preparation Complete!
Statistics:
Train:
  Real: 49,000 | Fake: 49,000 | Total: 98,000
Val:
  Real: 10,500 | Fake: 10,500 | Total: 21,000
Test:
  Real: 10,500 | Fake: 10,500 | Total: 21,000
```

---

## Alternative: Celeb-DF Dataset (Easier to Get)

If FaceForensics++ access is delayed, use Celeb-DF:

### Step 1: Download

1. **Visit**: https://github.com/yuezunli/celeb-deepfakeforensics
2. **Click**: "Request Dataset"
3. **Fill form**: Usually approved within hours
4. **Download**: Via provided Google Drive link

### Step 2: Structure

```
data/celebdf_raw/
├── Celeb-real/
│   ├── id0_0000.mp4
│   └── ...
└── Celeb-synthesis/
    ├── id0_id1_0000.mp4
    └── ...
```

### Step 3: Extract and Prepare

```bash
# Extract frames
python scripts/extract_frames_from_videos.py \
    --input data/celebdf_raw/Celeb-real \
    --output data/celebdf_frames/real \
    --fps 1

python scripts/extract_frames_from_videos.py \
    --input data/celebdf_raw/Celeb-synthesis \
    --output data/celebdf_frames/fake \
    --fps 1

# Prepare dataset
python scripts/prepare_dataset.py \
    --input data/celebdf_frames \
    --output data/processed \
    --resize
```

---

## Alternative: Kaggle DFDC Dataset (Requires Kaggle Account)

### Step 1: Setup Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Download your API key from Kaggle.com → Account → Create New API Token
# Place kaggle.json in: C:/Users/user/.kaggle/
```

### Step 2: Download

```bash
# Download dataset (warning: very large ~470GB)
kaggle competitions download -c deepfake-detection-challenge

# Or download sample (smaller, ~10GB)
kaggle datasets download -d mrmorj/deepfake-detection-sample
```

### Step 3: Extract and Prepare

```bash
# Extract
unzip deepfake-detection-sample.zip -d data/dfdc_raw

# Prepare
python scripts/prepare_dataset.py \
    --input data/dfdc_raw \
    --output data/processed
```

---

## What if I Can't Download Any Dataset?

### Option 1: Use Smaller Public Dataset

Try these immediately available datasets:

1. **140k Real and Fake Faces** (Kaggle - No approval needed)
   ```bash
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces
   unzip 140k-real-and-fake-faces.zip -d data/140k_faces
   ```

2. **Deepfake Detection Challenge Sample** (Kaggle - No approval needed)
   ```bash
   kaggle datasets download -d mrmorj/deepfake-detection-sample
   ```

### Option 2: Wait and Use Demo

While waiting for dataset access:
- Keep using the demo detector
- Test your UI/UX
- Deploy to Vercel
- Add other features

---

## Estimated Times

| Task | Time | Notes |
|------|------|-------|
| Request access | 5 min | Fill form |
| Wait for approval | 1-48 hours | Usually 24h |
| Download dataset | 1-4 hours | Depends on internet |
| Extract files | 30-60 min | Automatic |
| Extract frames (if video) | 1-2 hours | Automatic |
| Prepare dataset | 30-60 min | Automatic |
| **Total** | **4-6 hours** | Most is waiting/downloading |

---

## Next Steps After Download

Once `data/processed/` is ready:

```bash
# Start training
python backend/training/train_efficientnet.py \
    --data-dir data/processed \
    --batch-size 32 \
    --epochs 25 \
    --model efficientnet_b4
```

Training will take 4-8 hours on GPU, 2-3 days on CPU.

---

## Troubleshooting

### Can't get dataset access?
- Try Celeb-DF (faster approval)
- Use Kaggle public datasets (no approval needed)
- Use synthetic data for testing

### Download too slow?
- Use smaller dataset (Celeb-DF instead of full DFDC)
- Download overnight
- Ask friends with faster internet

### Not enough disk space?
- Need at least 50GB free
- Delete unnecessary files
- Use external hard drive

---

## Quick Start Command

After you've downloaded and extracted the dataset:

```bash
# One command to prepare everything
python scripts/prepare_dataset.py \
    --input data/YOUR_DATASET_HERE \
    --output data/processed \
    --resize \
    --target-size 224
```

Then you're ready to train! 🚀
