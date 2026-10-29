"""
Vercel Serverless Function: Noisy Audio Deepfake Detection
"""
import os
import sys
import uuid
import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        from backend.detectors.audio_detector import AudioDetector
        _detector = AudioDetector()
    return _detector

@app.post("/api/predict-noisy-audio")
async def predict_noisy_audio(file: UploadFile = File(...)):
    """Predict if an uploaded noisy audio is a deepfake"""
    temp_path = None
    clean_temp_path = None
    try:
        from backend.utils.noise_reduction_utils import reduce_noise_and_analyze
        from backend.models.audio_cnn import melspectrogram_from_file
        import scipy.io.wavfile as wavfile
        
        ext = Path(file.filename).suffix.lower()
        allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(allowed))}"
            )
        
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4()}{ext}"
        clean_temp_path = temp_dir / f"{uuid.uuid4()}_clean.wav"
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Apply noise reduction
        noise_result = reduce_noise_and_analyze(str(temp_path))
        
        # Save clean audio temporarily
        wavfile.write(
            str(clean_temp_path), 
            noise_result["sample_rate"], 
            (noise_result["clean_audio"] * 32767).astype('int16')
        )
        
        # Run deepfake detection on cleaned audio
        detector = get_detector()
        detection_result = detector.predict(str(clean_temp_path))
        
        # Adjust confidence based on noise level
        confidence_adjustment = 0
        if noise_result["noise_level"] == "High":
            confidence_adjustment = -15
        elif noise_result["noise_level"] == "Moderate":
            confidence_adjustment = -8
        
        adjusted_confidence = max(50, min(95, detection_result["confidence"] + confidence_adjustment))
        
        # Combine results
        result = {
            "verdict": detection_result["verdict"],
            "label": detection_result["verdict"],
            "probability": detection_result["probability"],
            "probability_pct": detection_result["probability_pct"],
            "confidence": round(adjusted_confidence, 1),
            "original_confidence": detection_result["confidence"],
            "confidence_adjustment": confidence_adjustment,
            "noise_analysis": {
                "original_snr": noise_result["original_snr"],
                "improved_snr": noise_result["improved_snr"],
                "snr_improvement": noise_result["snr_improvement"],
                "noise_level": noise_result["noise_level"],
                "noise_verdict": noise_result["noise_verdict"],
                "noise_reduction_quality": noise_result["noise_reduction_quality"]
            },
            "spectral_explanation": f"{noise_result['noise_verdict']} - {detection_result.get('spectral_explanation', 'Analyzed after noise reduction')}",
            "safety_recommendation": f"Audio quality: {noise_result['noise_level']} noise level. {detection_result.get('safety_recommendation', 'Verify through secondary channel.')}",
            "spectrogram_base64": detection_result.get("spectrogram_base64"),
            "features_analyzed": noise_result["features_analyzed"] + detection_result.get("features_analyzed", []),
            "filename": file.filename,
            "media_type": "noisy-audio",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass
        if clean_temp_path and clean_temp_path.exists():
            try:
                os.remove(clean_temp_path)
            except:
                pass

from mangum import Mangum
handler = Mangum(app, lifespan="off")
