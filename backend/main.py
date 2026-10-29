"""
FastAPI Backend Application for Deepfake Detection System.
Implements endpoints:
- GET  /health
- POST /predict-image
- POST /predict-video
- POST /predict-audio
- GET  /api/history
- POST /api/clear-history
- GET  /api/samples
- Static files for frontend UI.
"""
import os
import shutil
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import (
    BASE_DIR,
    ALLOWED_IMAGE_EXT,
    ALLOWED_VIDEO_EXT,
    ALLOWED_AUDIO_EXT,
    TEMP_UPLOAD_DIR,
    SAMPLE_DATA_DIR,
)
from backend.detectors.image_video_detector import ImageVideoDetector
from backend.detectors.audio_detector import AudioDetector

# Model selection
USE_MODEL = 'trained'  # Using YOUR trained EfficientNet-B4 (99.94% accuracy!)

app = FastAPI(
    title="Deepfake Detection System API",
    description="Advanced Deepfake Detection: ResNet18 + Forensic Analysis (Compression, Symmetry, Edges, Texture)",
    version="3.0.0"
)

# Enable CORS for seamless integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
print("Initializing detectors...")
try:
    if USE_MODEL == 'demo':
        print("Using Demo Detector (Varied results for prototyping)")
        from backend.detectors.demo_detector import DemoDeepfakeDetector
        image_video_detector = DemoDeepfakeDetector()
    elif USE_MODEL == 'production':
        print("Using Production-Calibrated ResNet50 Detector")
        from backend.detectors.production_detector import ProductionDeepfakeDetector
        image_video_detector = ProductionDeepfakeDetector()
    elif USE_MODEL == 'huggingface':
        print("Using HuggingFace ResNet18 + Advanced Forensic Analysis")
        from backend.detectors.huggingface_detector import HuggingFaceDeepfakeDetector
        image_video_detector = HuggingFaceDeepfakeDetector()
    elif USE_MODEL == 'multistrategy':
        print("Using Multi-Strategy Detector")
        from backend.detectors.pretrained_deepfake_detector import MultiStrategyDeepfakeDetector
        image_video_detector = MultiStrategyDeepfakeDetector()
    elif USE_MODEL == 'efficientnet':
        print("Using EfficientNet-B4 model")
        from backend.detectors.efficientnet_image_detector import EfficientNetImageDetector
        image_video_detector = EfficientNetImageDetector()
    elif USE_MODEL == 'trained':
        print("🎯 Using YOUR Trained EfficientNet-B4 (99.89% Val Accuracy)")
        from backend.detectors.trained_detector import TrainedDeepfakeDetector
        image_video_detector = TrainedDeepfakeDetector()
    elif USE_MODEL == 'working':
        print("✅ Using Working Forensic Detector (Production-Ready)")
        from backend.detectors.working_detector import WorkingDeepfakeDetector
        image_video_detector = WorkingDeepfakeDetector()
    else:
        print("Using GenD ViT model")
        image_video_detector = ImageVideoDetector()
    print("✓ Image/Video detector initialized successfully")
except Exception as e:
    print(f"✗ Error initializing image detector: {e}")
    import traceback
    traceback.print_exc()
    print("Falling back to basic detector...")
    image_video_detector = ImageVideoDetector()
    
try:
    audio_detector = AudioDetector()
    print("✓ AudioDetector initialized")
except Exception as e:
    print(f"✗ Error initializing AudioDetector: {e}")
    import traceback
    traceback.print_exc()

# In-memory history ledger (reflecting DFD Level 1 & UML Database store)
verification_history: List[Dict[str, Any]] = []

def _save_temp_file(upload: UploadFile, allowed_exts: set) -> Path:
    """Validates extension and stores uploaded file in temporary storage."""
    ext = Path(upload.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(allowed_exts))}"
        )
    temp_id = str(uuid.uuid4())
    temp_path = TEMP_UPLOAD_DIR / f"{temp_id}_{upload.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
    finally:
        upload.file.close()
    return temp_path

@app.get("/health")
def health_check():
    """Health check endpoint exposing system & detector status."""
    return {
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "device": image_video_detector.device,
        "detectors": {
            "visual_gend_vit": {
                "loaded": True,
                "weights_custom": image_video_detector.weights_loaded,
                "method": "GenD (LayerNorm-only fine-tuning + Hyperspherical Head)",
                "max_frames": image_video_detector.MAX_FRAMES
            },
            "audio_spectrogram_cnn": {
                "loaded": True,
                "weights_custom": audio_detector.weights_loaded,
                "method": "128-band Mel-Spectrogram 2D CNN Spoof Classifier"
            }
        }
    }

@app.post("/predict-image")
def predict_image(file: UploadFile = File(...)):
    """Routes image upload through Haar face detection and GenD ViT classifier."""
    temp_path = _save_temp_file(file, ALLOWED_IMAGE_EXT)
    try:
        result = image_video_detector.predict_image(str(temp_path))
        result["filename"] = file.filename
        result["media_type"] = "image"
        result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to audit history
        history_record = {
            "id": str(uuid.uuid4())[:8],
            "filename": file.filename,
            "media_type": "image",
            "verdict": result["verdict"],
            "probability_pct": result["probability_pct"],
            "confidence": result["confidence"],
            "timestamp": result["timestamp"]
        }
        verification_history.insert(0, history_record)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/predict-video")
def predict_video(file: UploadFile = File(...)):
    """Routes video upload through multi-frame face extraction and GenD ViT classifier."""
    temp_path = _save_temp_file(file, ALLOWED_VIDEO_EXT)
    try:
        print(f"[DEBUG] Starting video prediction for: {file.filename}")
        print(f"[DEBUG] Temp path: {temp_path}")
        
        result = image_video_detector.predict_video(str(temp_path))
        
        print(f"[DEBUG] Video prediction result: {result}")
        
        result["filename"] = file.filename
        result["media_type"] = "video"
        result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to audit history
        history_record = {
            "id": str(uuid.uuid4())[:8],
            "filename": file.filename,
            "media_type": "video",
            "verdict": result["verdict"],
            "probability_pct": result.get("average_probability_pct", result.get("probability_pct", 0)),
            "confidence": result["confidence"],
            "timestamp": result["timestamp"]
        }
        verification_history.insert(0, history_record)
        return result
    except Exception as e:
        print(f"[ERROR] Video prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/predict-audio")
def predict_audio(file: UploadFile = File(...)):
    """Routes audio clip through 16kHz resampling, mel spectrogram, and SpectrogramCNN."""
    temp_path = _save_temp_file(file, ALLOWED_AUDIO_EXT)
    try:
        result = audio_detector.predict(str(temp_path))
        result["filename"] = file.filename
        result["media_type"] = "audio"
        result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to audit history
        history_record = {
            "id": str(uuid.uuid4())[:8],
            "filename": file.filename,
            "media_type": "audio",
            "verdict": result["verdict"],
            "probability_pct": result["probability_pct"],
            "confidence": result["confidence"],
            "timestamp": result["timestamp"]
        }
        verification_history.insert(0, history_record)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.get("/api/history")
def get_history():
    """Returns past verification records."""
    return {"history": verification_history}

@app.post("/api/clear-history")
def clear_history():
    """Clears past verification history."""
    verification_history.clear()
    return {"status": "cleared"}

@app.get("/api/samples")
def get_samples():
    """Returns list of built-in sample test files available for 1-click evaluation."""
    samples = []
    if SAMPLE_DATA_DIR.exists():
        for f in SAMPLE_DATA_DIR.iterdir():
            if f.is_file():
                ext = f.suffix.lower()
                mtype = "image" if ext in ALLOWED_IMAGE_EXT else ("video" if ext in ALLOWED_VIDEO_EXT else ("audio" if ext in ALLOWED_AUDIO_EXT else "unknown"))
                samples.append({
                    "name": f.name,
                    "type": mtype,
                    "size_bytes": f.stat().st_size
                })
    return {"samples": samples}

@app.get("/api/load-sample/{filename}")
def load_sample_file(filename: str):
    """Executes prediction directly on a built-in sample file."""
    sample_file = SAMPLE_DATA_DIR / filename
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")

    ext = sample_file.suffix.lower()
    if ext in ALLOWED_IMAGE_EXT:
        res = image_video_detector.predict_image(str(sample_file))
        res["media_type"] = "image"
    elif ext in ALLOWED_VIDEO_EXT:
        res = image_video_detector.predict_video(str(sample_file))
        res["media_type"] = "video"
    elif ext in ALLOWED_AUDIO_EXT:
        res = audio_detector.predict(str(sample_file))
        res["media_type"] = "audio"
    else:
        raise HTTPException(status_code=400, detail="Unsupported sample format")

    res["filename"] = filename
    res["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return res

@app.post("/predict-blur")
def predict_blur_endpoint(file: UploadFile = File(...)):
    """Detects blur in uploaded image using Laplacian variance."""
    temp_path = _save_temp_file(file, ALLOWED_IMAGE_EXT)
    try:
        from backend.utils.image_quality_utils import detect_blur_laplacian
        result = detect_blur_laplacian(str(temp_path))
        
        # Format response
        response = {
            "filename": file.filename,
            "blur_score": result["laplacian_variance"],
            "is_blurry": result["verdict"] == "Blurry",
            "quality_assessment": result["quality"],
            "confidence": result["confidence"],
            "threshold": 100,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/predict-noisy-audio")
def predict_noisy_audio_endpoint(file: UploadFile = File(...)):
    """Routes noisy audio through noise reduction and deepfake detection."""
    temp_path = _save_temp_file(file, ALLOWED_AUDIO_EXT)
    try:
        import time
        from backend.utils.noise_reduction_utils import reduce_noise_and_analyze
        
        start_time = time.time()
        
        # Apply noise reduction
        noise_analysis = reduce_noise_and_analyze(str(temp_path))
        
        # Run deepfake detection on cleaned audio
        # For now, we'll save the clean audio temporarily and run prediction
        import tempfile
        import scipy.io.wavfile as wavfile
        
        clean_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        clean_temp_path = clean_temp.name
        clean_temp.close()
        
        # Save cleaned audio
        wavfile.write(clean_temp_path, noise_analysis["sample_rate"], 
                     (noise_analysis["clean_audio"] * 32768).astype('int16'))
        
        # Run deepfake prediction
        prediction_result = audio_detector.predict(clean_temp_path)
        
        # Clean up temporary file
        try:
            os.remove(clean_temp_path)
        except:
            pass
        
        processing_time = time.time() - start_time
        
        # Combine results
        response = {
            "filename": file.filename,
            "verdict": prediction_result["verdict"],
            "probability_pct": prediction_result["probability_pct"],
            "confidence": prediction_result["confidence"],
            "noise_reduced": True,
            "original_snr": noise_analysis["original_snr"],
            "improved_snr": noise_analysis["improved_snr"],
            "snr_improvement": noise_analysis["snr_improvement"],
            "noise_level": noise_analysis["noise_level"],
            "processing_time_sec": processing_time,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

# Static UI mount
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Deepfake Detection API is running. Place index.html inside frontend/ directory."}

@app.get("/image-analysis.html")
def serve_image_analysis():
    """Serve the image analysis tool page."""
    page_file = FRONTEND_DIR / "image-analysis.html"
    if page_file.exists():
        return FileResponse(page_file)
    raise HTTPException(status_code=404, detail="Image analysis page not found")

@app.get("/video-analysis.html")
def serve_video_analysis():
    """Serve the video analysis tool page."""
    page_file = FRONTEND_DIR / "video-analysis.html"
    if page_file.exists():
        return FileResponse(page_file)
    raise HTTPException(status_code=404, detail="Video analysis page not found")

@app.get("/audio-analysis.html")
def serve_audio_analysis():
    """Serve the audio analysis tool page."""
    page_file = FRONTEND_DIR / "audio-analysis.html"
    if page_file.exists():
        return FileResponse(page_file)
    raise HTTPException(status_code=404, detail="Audio analysis page not found")

@app.get("/blur-detection.html")
def serve_blur_detection():
    """Serve the blur detection tool page."""
    page_file = FRONTEND_DIR / "blur-detection.html"
    if page_file.exists():
        return FileResponse(page_file)
    raise HTTPException(status_code=404, detail="Blur detection page not found")

@app.get("/noisy-audio.html")
def serve_noisy_audio():
    """Serve the noisy audio detection tool page."""
    page_file = FRONTEND_DIR / "noisy-audio.html"
    if page_file.exists():
        return FileResponse(page_file)
    raise HTTPException(status_code=404, detail="Noisy audio page not found")

@app.get("/history.html")
def serve_history():
    """Serve the analysis history page."""
    page_file = FRONTEND_DIR / "history.html"
    if page_file.exists():
        return FileResponse(page_file)
    raise HTTPException(status_code=404, detail="History page not found")
