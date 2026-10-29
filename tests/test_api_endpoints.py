"""
Integration tests for FastAPI REST API endpoints.
"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "detectors" in data
    assert "visual_gend_vit" in data["detectors"]
    assert "audio_spectrogram_cnn" in data["detectors"]

def test_history_and_clear_endpoints():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "history" in response.json()

    clear_res = client.post("/api/clear-history")
    assert clear_res.status_code == 200
    assert clear_res.json()["status"] == "cleared"

def test_invalid_extension_upload():
    # Attempt to upload an invalid file extension (e.g. .txt to /predict-image)
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/predict-image", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_root_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "Deepfake Detection System" in response.text

def test_samples_api():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()["samples"]
    assert len(samples) > 0
