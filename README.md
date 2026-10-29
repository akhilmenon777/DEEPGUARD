# 🛡️ DeepGuard - AI-Powered Deepfake Detection System

<div align="center">

![DeepGuard Banner](https://img.shields.io/badge/DeepGuard-AI%20Detection-F26B4F?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-99.94%25-success?style=for-the-badge)

**Advanced Multi-Modal Deepfake Detection using State-of-the-Art Deep Learning**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📖 Overview

**DeepGuard** is a production-ready deepfake detection system that combines computer vision and audio analysis to identify AI-generated content. Built with EfficientNet-B4 and custom CNN architectures, achieving **99.94% validation accuracy** on real-world datasets.

### 🎯 Key Highlights

- ✅ **Multi-Modal Detection**: Image, Video, and Audio analysis
- ✅ **State-of-the-Art Accuracy**: 99.94% on 140K Real and Fake Faces dataset
- ✅ **Production-Ready**: FastAPI backend with professional web interface
- ✅ **GPU Accelerated**: CUDA support for real-time inference
- ✅ **Robust**: Handles blur, noise, and quality variations
- ✅ **Complete Pipeline**: Training, evaluation, and deployment code included

---

## 🚀 Features

### 🖼️ Image Analysis
- Face detection and extraction using OpenCV
- EfficientNet-B4 backbone with custom classifier
- Heatmap visualization of suspicious regions
- Confidence scoring with calibrated thresholds

### 🎬 Video Analysis
- Frame-by-frame analysis (30 frames per video)
- Temporal aggregation for robust predictions
- Support for MP4, AVI, MOV, MKV formats
- Per-frame confidence tracking

### 🎤 Audio Deepfake Detection
- Custom CNN architecture for mel-spectrogram analysis
- Voice cloning and TTS detection
- Noise-resistant preprocessing
- Support for WAV, MP3, FLAC formats

### 🔍 Quality Assessment
- Blur detection using Laplacian variance
- Noisy audio handling with signal enhancement
- Robust to real-world conditions

### 📊 Analysis History
- Complete audit trail of all detections
- Filter by media type and verdict
- Export capabilities for reporting
- Timestamp tracking

---

## 🎨 Tech Stack

### Backend
- **Framework**: FastAPI (async, high-performance)
- **Deep Learning**: PyTorch 2.0+, timm library
- **Computer Vision**: OpenCV, PIL
- **Audio Processing**: scipy, librosa

### Frontend
- **Design**: Peach Voltage theme (professional UI)
- **Framework**: Vanilla JavaScript (no dependencies)
- **Styling**: Modern CSS with gradients and animations

### Model Architecture
- **Backbone**: EfficientNet-B4 (pre-trained on ImageNet)
- **Classifier**: 3-layer MLP (1792 → 512 → 256 → 2)
- **Loss Function**: Focal Loss for hard example mining
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

---

## 📊 Results

### Training Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 99.94% |
| **Real Class Accuracy** | 99.98% |
| **Fake Class Accuracy** | 99.87% |
| **Training Time** | ~6 hours (NVIDIA RTX 3050) |
| **Dataset Size** | 140,000 images |

### Dataset Details
- **Source**: [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Split**: 100K train / 20K validation / 20K test
- **Balance**: 50% real, 50% fake (perfectly balanced)
- **Augmentation**: Random flips, rotations, color jitter

---

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM
- 2GB+ GPU VRAM (recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/deepguard.git
cd deepguard
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv_gpu
# Windows
venv_gpu\Scripts\activate
# Linux/Mac
source venv_gpu/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights
**Option A**: Train from scratch (see [Training](#-training))

**Option B**: Download pre-trained weights
```bash
# Create weights directory
mkdir -p backend/weights

# Download weights (link to be provided)
# Place efficientnet_b4_best.pth in backend/weights/
```

---

## 🚀 Usage

### Start the Server
```bash
python serve.py
```

### Access the Web Interface
Open your browser and navigate to:
```
http://localhost:8001
```

### API Endpoints

#### Image Analysis
```bash
curl -X POST http://localhost:8001/predict-image \
  -F "file=@path/to/image.jpg"
```

#### Video Analysis
```bash
curl -X POST http://localhost:8001/predict-video \
  -F "file=@path/to/video.mp4"
```

#### Audio Analysis
```bash
curl -X POST http://localhost:8001/predict-audio \
  -F "file=@path/to/audio.wav"
```

---

## 🎓 Training

### Step 1: Download Dataset
```bash
# Using Kaggle API
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip 140k-real-and-fake-faces.zip -d data/140k_faces/
```

### Step 2: Train the Model
```bash
python backend/training/train_balanced.py \
  --data-dir data/140k_faces/real_vs_fake/real-vs-fake \
  --batch-size 32 \
  --epochs 10 \
  --lr 0.0001
```

### Training Configuration
- **Batch Size**: 32 (adjust based on GPU memory)
- **Learning Rate**: 1e-4 with ReduceLROnPlateau
- **Epochs**: 10 (early stopping enabled)
- **Loss**: Focal Loss (gamma=2.0, alpha=0.25)
- **Augmentation**: Random flips, crops, color jitter

---

## 🏗️ Architecture

### System Overview
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Frontend  │─────▶│  FastAPI     │─────▶│  PyTorch    │
│   (HTML/JS) │      │  Backend     │      │  Model      │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Face/Audio  │
                     │  Extraction  │
                     └──────────────┘
```

### Model Architecture
```
Input Image (224x224x3)
    │
    ▼
EfficientNet-B4 Backbone
    │ (1792 features)
    ▼
Dropout(0.3)
    │
    ▼
Linear(1792 → 512) + ReLU + BatchNorm
    │
    ▼
Dropout(0.15)
    │
    ▼
Linear(512 → 256) + ReLU + BatchNorm
    │
    ▼
Dropout(0.15)
    │
    ▼
Linear(256 → 2)
    │
    ▼
Softmax → [Real Prob, Fake Prob]
```

---

## 📁 Project Structure

```
deepguard/
├── backend/
│   ├── detectors/
│   │   ├── trained_detector.py      # Main image/video detector
│   │   └── audio_detector.py        # Audio deepfake detector
│   ├── models/
│   │   ├── gend_vit.py             # Vision Transformer (alternative)
│   │   └── audio_cnn.py            # Audio CNN architecture
│   ├── training/
│   │   ├── train_balanced.py       # Training script
│   │   ├── dataset.py              # Dataset loader
│   │   └── losses.py               # Focal Loss implementation
│   ├── utils/
│   │   ├── face_utils.py           # Face detection utilities
│   │   └── audio_utils.py          # Audio preprocessing
│   ├── weights/                    # Model checkpoints (.pth)
│   └── main.py                     # FastAPI application
├── frontend/
│   ├── index.html                  # Landing page
│   ├── image-analysis.html         # Image detection UI
│   ├── video-analysis.html         # Video detection UI
│   ├── audio-analysis.html         # Audio detection UI
│   ├── history.html               # Audit trail
│   └── css/
│       └── deepguard-theme.css    # Peach Voltage theme
├── sample_data/                   # Demo files
├── requirements.txt               # Python dependencies
├── serve.py                      # Server startup script
└── README.md
```

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Test Files Included
- `test_gend_model.py` - Model architecture tests
- `test_audio_cnn.py` - Audio model tests
- `test_api_endpoints.py` - API endpoint tests
- `test_dataset_140k.py` - Dataset loading tests

---

## 🔒 Security & Privacy

- ✅ All processing happens locally (no data sent to external servers)
- ✅ Temporary files are automatically cleaned up
- ✅ No user data is stored permanently
- ✅ CORS enabled for controlled access
- ✅ Input validation and sanitization

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{deepguard2024,
  title={DeepGuard: Multi-Modal Deepfake Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/YOUR_USERNAME/deepguard}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) by xhlulu
- **EfficientNet**: [timm library](https://github.com/huggingface/pytorch-image-models) by Ross Wightman
- **Inspiration**: Academic research in deepfake detection and computer vision

---

## 📧 Contact

**Project Link**: [https://github.com/YOUR_USERNAME/deepguard](https://github.com/YOUR_USERNAME/deepguard)

**Your Name** - your.email@example.com

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ and PyTorch

</div>
