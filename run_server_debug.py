"""
Debug launcher for the Deepfake Detection System FastAPI server.
Uses port 8001 to avoid conflicts.
"""
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 65)
    print(" AI Deepfake Detection System Server - DEBUG MODE")
    print(" Architecture: GenD ViT (LayerNorm tuning) + Mel-Spectrogram CNN")
    print("=" * 65)
    print(" Dashboard URL: http://localhost:8001")
    print(" Swagger Docs:  http://localhost:8001/docs")
    print("=" * 65)
    print("\nPress CTRL+C to stop the server\n")

    try:
        uvicorn.run(
            "backend.main:app", 
            host="127.0.0.1", 
            port=8001, 
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
    except Exception as e:
        print(f"\n\n✗ Server error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
