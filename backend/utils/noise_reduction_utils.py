"""
Noise reduction and audio preprocessing utilities for noisy audio detection.
"""
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from typing import Tuple, Dict, Any


def estimate_noise_profile(audio_signal: np.ndarray, sr: int, noise_duration: float = 0.5) -> np.ndarray:
    """
    Estimates noise profile from the beginning of the audio signal.
    
    Args:
        audio_signal: Audio signal array
        sr: Sample rate
        noise_duration: Duration in seconds to use for noise estimation
        
    Returns:
        Noise profile spectrum
    """
    noise_samples = int(sr * noise_duration)
    noise_segment = audio_signal[:min(noise_samples, len(audio_signal))]
    
    # Compute noise spectrum using FFT
    noise_fft = np.fft.rfft(noise_segment)
    noise_profile = np.abs(noise_fft)
    
    return noise_profile


def spectral_subtraction(audio_signal: np.ndarray, noise_profile: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """
    Performs spectral subtraction to reduce noise.
    
    Args:
        audio_signal: Noisy audio signal
        noise_profile: Noise spectrum profile
        alpha: Oversubtraction factor
        
    Returns:
        Noise-reduced audio signal
    """
    # Compute signal spectrum
    signal_fft = np.fft.rfft(audio_signal)
    signal_magnitude = np.abs(signal_fft)
    signal_phase = np.angle(signal_fft)
    
    # Extend noise profile to match signal length if needed
    if len(noise_profile) < len(signal_magnitude):
        noise_profile = np.pad(noise_profile, (0, len(signal_magnitude) - len(noise_profile)), mode='edge')
    elif len(noise_profile) > len(signal_magnitude):
        noise_profile = noise_profile[:len(signal_magnitude)]
    
    # Spectral subtraction
    clean_magnitude = signal_magnitude - alpha * noise_profile
    clean_magnitude = np.maximum(clean_magnitude, 0.1 * signal_magnitude)  # Spectral floor
    
    # Reconstruct signal
    clean_fft = clean_magnitude * np.exp(1j * signal_phase)
    clean_signal = np.fft.irfft(clean_fft, n=len(audio_signal))
    
    return clean_signal.astype(np.float32)


def calculate_snr(signal: np.ndarray, noise_profile: np.ndarray) -> float:
    """
    Calculates Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        signal: Audio signal
        noise_profile: Noise spectrum
        
    Returns:
        SNR in decibels
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise_profile ** 2)
    
    if noise_power == 0:
        return 100.0  # Very high SNR
    
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def reduce_noise_and_analyze(audio_path: str, target_sr: int = 16000, duration: float = 4.0) -> Dict[str, Any]:
    """
    Reduces noise from audio file and provides analysis.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        duration: Duration in seconds
        
    Returns:
        Dictionary containing noise-reduced audio and analysis metrics
    """
    # Load audio
    try:
        sr, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
    except Exception:
        # Fallback
        data = np.zeros(int(target_sr * duration), dtype=np.float32)
        sr = target_sr
    
    # Resample if needed
    if sr != target_sr:
        num_resampled = int(len(data) * target_sr / sr)
        data = signal.resample(data, num_resampled)
        sr = target_sr
    
    # Trim or pad
    target_samples = int(sr * duration)
    if len(data) > target_samples:
        data = data[:target_samples]
    elif len(data) < target_samples:
        data = np.pad(data, (0, target_samples - len(data)), mode='constant')
    
    # Original SNR estimate
    noise_profile = estimate_noise_profile(data, sr)
    original_snr = calculate_snr(data, noise_profile)
    
    # Apply noise reduction
    clean_audio = spectral_subtraction(data, noise_profile, alpha=2.0)
    
    # Calculate improved SNR
    improved_snr = calculate_snr(clean_audio, noise_profile)
    snr_improvement = improved_snr - original_snr
    
    # Determine noise level
    if original_snr > 20:
        noise_level = "Low"
        noise_verdict = "Clean Audio"
    elif original_snr > 10:
        noise_level = "Moderate"
        noise_verdict = "Noisy Audio"
    else:
        noise_level = "High"
        noise_verdict = "Very Noisy Audio"
    
    # Quality metrics
    noise_reduction_quality = min(100, max(0, (improved_snr / 30) * 100))
    
    return {
        "clean_audio": clean_audio,
        "sample_rate": sr,
        "original_snr": round(original_snr, 2),
        "improved_snr": round(improved_snr, 2),
        "snr_improvement": round(snr_improvement, 2),
        "noise_level": noise_level,
        "noise_verdict": noise_verdict,
        "noise_reduction_quality": round(noise_reduction_quality, 1),
        "features_analyzed": [
            "Spectral subtraction noise reduction",
            "Signal-to-Noise Ratio (SNR) calculation",
            f"Noise level: {noise_level}",
            f"SNR improvement: +{round(snr_improvement, 1)} dB"
        ]
    }
