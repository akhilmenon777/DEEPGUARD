"""
Vercel Serverless Function: Audio Deepfake Detection
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

@app.post("/api/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    """Predict if an uploaded audio is a deepfake"""
    temp_path = None
    try:
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
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        detector = get_detector()
        result = detector.predict(str(temp_path))
        
        result["filename"] = file.filename
        result["media_type"] = "audio"
        result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass

from mangum import Mangum
handler = Mangum(app, lifespan="off")
