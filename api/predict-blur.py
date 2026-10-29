"""
Vercel Serverless Function: Image Blur Detection
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

@app.post("/api/predict-blur")
async def predict_blur(file: UploadFile = File(...)):
    """Detect blur and image quality issues"""
    temp_path = None
    try:
        from backend.utils.image_quality_utils import detect_blur_laplacian, extract_image_features
        from backend.utils.face_utils import numpy_to_base64_png
        
        ext = Path(file.filename).suffix.lower()
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
        
        # Blur detection
        blur_result = detect_blur_laplacian(str(temp_path))
        
        # Feature extraction
        features = extract_image_features(str(temp_path))
        
        # Encode images to base64
        original_base64 = numpy_to_base64_png(blur_result["original_image"])
        edges_base64 = numpy_to_base64_png(blur_result["edges_image"])
        
        result = {
            "verdict": blur_result["verdict"],
            "label": blur_result["verdict"],
            "quality": blur_result["quality"],
            "confidence": blur_result["confidence"],
            "laplacian_variance": blur_result["laplacian_variance"],
            "sharpness_score": blur_result["sharpness_score"],
            "high_freq_ratio": blur_result["high_freq_ratio"],
            "probability": (100 - blur_result["sharpness_score"]) / 100,
            "probability_pct": round(100 - blur_result["sharpness_score"], 1),
            "original_image_base64": original_base64,
            "edges_image_base64": edges_base64,
            "image_features": features,
            "features_analyzed": blur_result["features_analyzed"],
            "filename": file.filename,
            "media_type": "blur",
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

from mangum import Mangum
handler = Mangum(app, lifespan="off")
