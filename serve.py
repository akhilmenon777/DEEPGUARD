"""
Alternative server launcher using uvicorn directly without subprocess issues.
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 70)
    print(" AI Deepfake Detection System")
    print(" GenD ViT + Mel-Spectrogram CNN")
    print("=" * 70)
    print(" Server URL: http://localhost:8001")
    print(" API Docs:   http://localhost:8001/docs")
    print("=" * 70)
    print("\n Starting server... Press CTRL+C to stop\n")
    
    import uvicorn
    
    config = uvicorn.Config(
        "backend.main:app",
        host="127.0.0.1",
        port=8001,
        log_level="info",
        access_log=True,
        reload=False,
        workers=1
    )
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n\n Server stopped by user (CTRL+C)")
    except Exception as e:
        print(f"\n\n ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n Server has shut down")
        input("\n Press Enter to close this window...")
