# Deepfake Detection System - Vercel Edition 🛡️

A serverless deepfake detection system powered by GenD Vision Transformer and Audio CNN, deployable to Vercel.

## 🌟 Features

- **Image Detection**: Analyze images for deepfake manipulation using GenD ViT
- **Video Detection**: Multi-frame analysis with temporal consistency checking
- **Audio Detection**: Voice cloning and synthetic speech detection
- **Grad-CAM Visualization**: Explainable AI with attention heatmaps
- **Serverless Architecture**: Scalable deployment on Vercel
- **Modern UI**: Beautiful glassmorphism design with real-time results

## 🚀 Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd deepfake-detection-system
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r api/requirements.txt
   ```

3. **Run locally with Vercel CLI**:
   ```bash
   npm install -g vercel
   vercel dev
   ```

4. **Open in browser**:
   ```
   http://localhost:3000
   ```

### Deploy to Vercel

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Import to Vercel**:
   - Visit [vercel.com/new](https://vercel.com/new)
   - Import your GitHub repository
   - Click "Deploy"

3. **Done!** Your app is live at `https://your-project.vercel.app`

## 📁 Project Structure

```
deepfake-detection-system/
├── api/                      # Serverless API endpoints
│   ├── index.py             # Health check
│   ├── predict-image.py     # Image detection
│   ├── predict-video.py     # Video detection
│   ├── predict-audio.py     # Audio detection
│   └── requirements.txt     # Python dependencies
├── backend/                  # Core detection logic
│   ├── detectors/           # Detection implementations
│   ├── models/              # Neural network models
│   └── utils/               # Helper utilities
├── frontend/                 # Web interface
│   ├── index.html           # Main page
│   ├── css/style.css        # Styles
│   └── js/app.js            # Client logic
├── weights/                  # Model weights (see notes)
└── vercel.json              # Vercel configuration
```

## 🎯 API Endpoints

### `GET /api/health`
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "service": "deepfake-detection-api",
  "version": "2.0.0"
}
```

### `POST /api/predict-image`
Detect deepfakes in images

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "verdict": "Authentic",
  "probability": 0.123,
  "probability_pct": 12.3,
  "confidence": 87.7,
  "face_detected": true,
  "heatmap_base64": "data:image/png;base64,...",
  "face_crop_base64": "data:image/png;base64,...",
  "timestamp": "2024-10-21 14:30:00"
}
```

### `POST /api/predict-video`
Detect deepfakes in videos

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "verdict": "Likely Deepfake",
  "average_probability": 0.756,
  "average_probability_pct": 75.6,
  "confidence": 75.6,
  "frames_analyzed": 12,
  "duration_sec": 10.5,
  "frame_timeline": [...],
  "most_suspicious_frame": {...}
}
```

### `POST /api/predict-audio`
Detect synthetic audio and voice cloning

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "verdict": "Potentially Synthetic",
  "probability": 0.652,
  "probability_pct": 65.2,
  "confidence": 65.2,
  "spectral_explanation": "...",
  "safety_recommendation": "...",
  "spectrogram_base64": "data:image/png;base64,..."
}
```

## 🧠 Models

### GenD Vision Transformer
- **Architecture**: ViT Tiny Patch16 (224×224)
- **Training**: LayerNorm-only fine-tuning (~0.03% parameters)
- **Projection**: L2-normalized hyperspherical embedding
- **Loss**: Cross-Entropy + Supervised Contrastive Loss

### Audio Spectrogram CNN
- **Input**: 128-band Mel-spectrogram (16kHz, 4s)
- **Architecture**: 3-block 2D CNN
- **Features**: High-frequency energy analysis, temporal consistency

## ⚙️ Configuration

### Function Settings (vercel.json)

```json
{
  "functions": {
    "api/**/*.py": {
      "memory": 3008,
      "maxDuration": 60
    }
  }
}
```

### Environment Variables

Add in Vercel Dashboard if needed:
- `MODEL_WEIGHTS_URL`: External URL for model weights
- `MAX_FRAMES`: Maximum frames to analyze in videos (default: 20)

## 📊 Model Weights

**Important**: Model weights (`.pth` files) may exceed Vercel's deployment size limits.

### Options:

1. **External Storage** (Recommended):
   - Upload weights to S3/Cloud Storage
   - Download at runtime (first request)
   - See `VERCEL_DEPLOYMENT.md` for details

2. **Git LFS**:
   - Use Git Large File Storage
   - Push weights to repository
   - Vercel will clone them

3. **Smaller Models**:
   - Use quantized models
   - Train smaller variants

## 🔒 Security

- **Input Validation**: File type and size checks
- **CORS**: Configured for cross-origin requests
- **Rate Limiting**: Consider adding for production
- **Sanitization**: All user inputs are validated

## 🎨 Frontend Features

- **Drag & Drop**: Easy file upload
- **Live Preview**: See uploaded media before analysis
- **Real-time Results**: Instant verdict display
- **Grad-CAM Heatmaps**: Visual explanation of detections
- **Timeline View**: Frame-by-frame video analysis
- **Spectrogram View**: Audio frequency analysis
- **Audit History**: Track all verifications
- **Export Reports**: Download forensic reports

## 🐛 Troubleshooting

### Function Timeout
- Reduce `MAX_FRAMES` for videos
- Increase `maxDuration` in vercel.json (Pro tier)

### Memory Issues
- Use smaller models
- Reduce batch sizes
- Clear GPU cache after inference

### Cold Starts
- First request after idle is slow (1-5s)
- Consider Vercel Pro for faster warm-up
- Use health check pings to keep warm

## 📈 Performance

- **Image Analysis**: ~2-4 seconds
- **Video Analysis**: ~5-15 seconds (depends on frames)
- **Audio Analysis**: ~2-3 seconds
- **Cold Start**: +1-5 seconds (first request)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `vercel dev`
5. Submit a pull request

## 📄 License

See LICENSE file for details.

## 👥 Authors

**Project Team**:
- Yash Jain (K019)
- Akhil Menon (K036)
- Taaksh Patel (K042)

## 📚 References

- **GenD Paper**: "Deepfake Detection that Generalizes Across Benchmarks", arXiv:2508.06248
- **ViT Paper**: "An Image is Worth 16x16 Words", ICLR 2021
- **SupCon**: "Supervised Contrastive Learning", NeurIPS 2020

## 🔗 Links

- [Live Demo](https://your-project.vercel.app) (after deployment)
- [Deployment Guide](VERCEL_DEPLOYMENT.md)
- [Original README](README.md)
- [Vercel Documentation](https://vercel.com/docs)

## ⭐ Features Coming Soon

- [ ] Batch processing
- [ ] Real-time video streaming analysis
- [ ] Mobile app
- [ ] API authentication
- [ ] Rate limiting
- [ ] Webhook notifications

---

**Deploy now and start detecting deepfakes!** 🚀

```bash
vercel --prod
```
