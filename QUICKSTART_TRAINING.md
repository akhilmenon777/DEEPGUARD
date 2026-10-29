# Quick Start: Train Your Model in 3 Steps

This guide gets you started with training in under 30 minutes using a test dataset.

## Step 1: Generate Test Dataset (5 minutes)

```bash
python scripts/generate_sample_dataset.py --num-samples 500
```

This creates a small synthetic dataset for testing:
- **1,000 images total** (500 real + 500 fake)
- Split into train/val/test
- Location: `data/sample/`

## Step 2: Train Model (15 minutes)

```bash
python backend/training/train_efficientnet.py \
    --data-dir data/sample \
    --batch-size 16 \
    --epochs 10 \
    --model efficientnet_b0
```

Training will show progress:
```
Epoch 1/10: 100%|████████| 43/43 [01:23<00:00, 1.9s/it]
Train Loss: 0.512 | Train Acc: 72.3%
Val Loss: 0.428 | Val Acc: 78.5%
✓ Saved best model to backend/weights/efficientnet_b0_best.pth
```

## Step 3: Use Trained Model (2 minutes)

Update `backend/main.py`:

```python
USE_MODEL = 'trained'
```

Then create `backend/detectors/trained_detector.py`:

```python
from backend.detectors.efficientnet_image_detector import EfficientNetImageDetector

class TrainedDetector(EfficientNetImageDetector):
    def __init__(self, device=None):
        super().__init__(device)
        # Load your trained weights
        import torch
        weights = torch.load('backend/weights/efficientnet_b0_best.pth')
        self.model.load_state_dict(weights['model_state_dict'])
```

Restart server:
```bash
python serve.py
```

**Done!** Your model is now trained and running.

---

## Next: Train on Real Data

For production accuracy, use real datasets:

### Option A: FaceForensics++ (Recommended)

1. **Register and download**: https://github.com/ondyari/FaceForensics
2. **Extract to**: `data/faceforensics/`
3. **Prepare dataset**:
   ```bash
   python scripts/prepare_dataset.py \
       --input data/faceforensics \
       --output data/processed
   ```
4. **Train**:
   ```bash
   python backend/training/train_efficientnet.py \
       --data-dir data/processed \
       --batch-size 32 \
       --epochs 25 \
       --model efficientnet_b4
   ```

Expected time: 4-8 hours on GPU
Expected accuracy: 93-96%

### Option B: Download Pre-trained Weights

If you don't have time to train:

1. **Kaggle DFDC Competition**: https://www.kaggle.com/c/deepfake-detection-challenge
2. Download winning solution weights
3. Convert and use in your detector

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
--batch-size 8
```

### "No samples found"
Check directory structure:
```
data/sample/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Training too slow
Use smaller model:
```bash
--model efficientnet_b0  # Instead of b4
```

---

## Performance Comparison

| Dataset | Size | Training Time | Accuracy |
|---------|------|---------------|----------|
| Sample (synthetic) | 1K | 15 min | ~75% |
| Small custom | 10K | 1-2 hours | ~85% |
| FaceForensics++ | 140K | 4-8 hours | ~94% |
| DFDC | 120K videos | 8-12 hours | ~96% |

---

## What's Next?

After training:

1. ✅ Test model accuracy
2. ✅ Try different images
3. ✅ Deploy to Vercel
4. ✅ Share your results!

Questions? Check `TRAINING_GUIDE.md` for detailed instructions.
