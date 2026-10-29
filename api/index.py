"""
Main Vercel Serverless API Entry Point
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="Deepfake Detection API",
    description="Serverless deepfake detection using GenD ViT and Audio CNN",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")
@app.get("/api/")
def root():
    """API Root Endpoint"""
    return {
        "message": "Deepfake Detection API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/api/health",
            "predict_image": "/api/predict-image",
            "predict_video": "/api/predict-video",
            "predict_audio": "/api/predict-audio"
        }
    }

@app.get("/api/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "deepfake-detection-api",
        "version": "2.0.0"
    }

# Handler for Vercel
def handler(request, context=None):
    """Vercel serverless function handler"""
    from mangum import Mangum
    asgi_handler = Mangum(app, lifespan="off")
    return asgi_handler(request, context)
