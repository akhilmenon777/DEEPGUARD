"""
Launcher script for the Deepfake Detection System FastAPI server.
"""
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 65)
    print(" AI Deepfake Detection System Server")
    print(" Architecture: GenD ViT (LayerNorm tuning) + Mel-Spectrogram CNN")
    print(" SE Activity / Project: Yash Jain, Akhil Menon, Taaksh Patel")
    print("=" * 65)
    print(" Dashboard URL: http://localhost:8000")
    print(" Swagger Docs:  http://localhost:8000/docs")
    print("=" * 65)

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
