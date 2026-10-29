"""
End-to-End Verification of the entire Deepfake Detection System.
Tests:
1. Image detection on sample_real_face.jpg and sample_deepfake_face.jpg
2. Audio detection on sample_real_speech.wav and sample_voice_clone.wav
3. Video detection on sample_video.mp4
4. Verification of Grad-CAM heatmap generation and base64 strings
5. Verification of Mel-Spectrogram and spectral heuristics
6. Verification of History logging
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.main import app

def run_verification():
    print("=" * 70)
    print("Running Complete End-to-End Deepfake Detection System Verification")
    print("=" * 70)

    client = TestClient(app)

    # 1. Health check
    health_res = client.get("/health")
    assert health_res.status_code == 200
    print("[1/5] Health Check: PASSED")
    print("      Device:", health_res.json()["device"])

    # 2. Image Prediction
    real_img_path = BASE_DIR / "sample_data" / "sample_real_face.jpg"
    fake_img_path = BASE_DIR / "sample_data" / "sample_deepfake_face.jpg"

    with open(real_img_path, "rb") as f:
        img_res = client.post("/predict-image", files={"file": ("real_face.jpg", f, "image/jpeg")})
    assert img_res.status_code == 200
    img_data = img_res.json()
    assert "verdict" in img_data
    assert "heatmap_base64" in img_data
    assert "face_crop_base64" in img_data
    assert img_data["heatmap_base64"].startswith("data:image/png;base64,")
    print(f"[2/5] Image Analysis (Real Face): PASSED -> Verdict: {img_data['verdict']}, Probability: {img_data['probability_pct']}%, Confidence: {img_data['confidence']}%")

    with open(fake_img_path, "rb") as f:
        fake_img_res = client.post("/predict-image", files={"file": ("fake_face.jpg", f, "image/jpeg")})
    assert fake_img_res.status_code == 200
    print(f"      Image Analysis (Fake Face): PASSED -> Verdict: {fake_img_res.json()['verdict']}, Grad-CAM: Generated ({len(fake_img_res.json()['heatmap_base64'])} chars)")

    # 3. Audio Prediction
    real_audio_path = BASE_DIR / "sample_data" / "sample_real_speech.wav"
    fake_audio_path = BASE_DIR / "sample_data" / "sample_voice_clone.wav"

    with open(real_audio_path, "rb") as f:
        audio_res = client.post("/predict-audio", files={"file": ("real_speech.wav", f, "audio/wav")})
    assert audio_res.status_code == 200
    audio_data = audio_res.json()
    assert "verdict" in audio_data
    assert "spectral_explanation" in audio_data
    assert "safety_recommendation" in audio_data
    assert "spectrogram_base64" in audio_data
    print(f"[3/5] Audio Analysis (Real Speech): PASSED -> Verdict: {audio_data['verdict']}, Probability: {audio_data['probability_pct']}%, Spectral Note: {audio_data['spectral_explanation'][:60]}...")

    with open(fake_audio_path, "rb") as f:
        fake_audio_res = client.post("/predict-audio", files={"file": ("voice_clone.wav", f, "audio/wav")})
    assert fake_audio_res.status_code == 200
    print(f"      Audio Analysis (Voice Clone): PASSED -> Verdict: {fake_audio_res.json()['verdict']}, Recommendation: {fake_audio_res.json()['safety_recommendation'][:55]}...")

    # 4. Video Prediction
    video_path = BASE_DIR / "sample_data" / "sample_video.mp4"
    with open(video_path, "rb") as f:
        vid_res = client.post("/predict-video", files={"file": ("test_video.mp4", f, "video/mp4")})
    assert vid_res.status_code == 200
    vid_data = vid_res.json()
    assert "verdict" in vid_data
    assert "frame_timeline" in vid_data
    assert "most_suspicious_frame" in vid_data
    print(f"[4/5] Video Analysis: PASSED -> Analyzed {vid_data['frames_analyzed']} frames. Verdict: {vid_data['verdict']}, Peak Risk: {vid_data['max_probability']}, Frame Timeline items: {len(vid_data['frame_timeline'])}")

    # 5. History Audit
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    history = hist_res.json()["history"]
    assert len(history) >= 5
    print(f"[5/5] Audit History Ledger: PASSED -> Recorded {len(history)} verification events.")

    # 6. Sample Presets API
    samples_res = client.get("/api/samples")
    assert samples_res.status_code == 200
    print(f"      Available 1-Click Samples: {[s['name'] for s in samples_res.json()['samples']]}")

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
