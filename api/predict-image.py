"""
Vercel Serverless Function: Image Deepfake Detection
"""
import os
import sys
import uuid
import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load detector to optimize cold starts
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        from backend.detectors.image_video_detector import ImageVideoDetector
        _detector = ImageVideoDetector()
    return _detector

@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Predict if an uploaded image is a deepfake"""
    temp_path = None
    try:
        # Validate file extension
        ext = Path(file.filename).suffix.lower()
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(allowed))}"
            )
        
        # Save to temp
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4()}{ext}"
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run detection
        detector = get_detector()
        result = detector.predict_image(str(temp_path))
        
        result["filename"] = file.filename
        result["media_type"] = "image"
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

# Vercel handler
from mangum import Mangum
handler = Mangum(app, lifespan="off")
