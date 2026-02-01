"""
AI Voice Detection API - FIXED VERSION
Fixes 422 Unprocessable Entity error
Accepts any request format the hackathon tester sends
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
import librosa
import numpy as np
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Voice Detection API",
    description="AI-generated voice detection supporting 5 Indian languages",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "hackathon_2026_voice_detection_secure_key"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


# ============================================================
# AUDIO FEATURE EXTRACTION
# ============================================================

def extract_features(audio_data: bytes):
    """Extract audio features using librosa"""
    y, sr = librosa.load(io.BytesIO(audio_data), sr=None)

    features = {}

    # Spectral
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["centroid_mean"] = float(np.mean(spec_cent))
    features["centroid_std"] = float(np.std(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["bandwidth_mean"] = float(np.mean(spec_bw))
    features["bandwidth_std"] = float(np.std(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["rolloff_mean"] = float(np.mean(rolloff))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["contrast_std"] = float(np.std(contrast))

    # Temporal
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    rms = librosa.feature.rms(y=y)[0]
    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))
    features["energy_variance"] = float(np.var(rms))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features["mfcc_0_std"] = float(np.std(mfccs[0]))
    features["mfcc_1_std"] = float(np.std(mfccs[1]))

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        if p > 0:
            pitch_values.append(p)

    if len(pitch_values) > 10:
        features["pitch_mean"] = float(np.mean(pitch_values))
        features["pitch_std"] = float(np.std(pitch_values))
        features["pitch_range"] = float(np.max(pitch_values) - np.min(pitch_values))
    else:
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_range"] = 0.0

    features["duration"] = float(len(y) / sr)

    return features


# ============================================================
# AI VOICE DETECTION ALGORITHM
# ============================================================

def detect_ai_voice(features, language):
    """Detect if voice is AI or Human based on audio features"""

    ai_score = 0.0
    human_score = 0.0
    ai_indicators = []
    human_indicators = []

    # Pitch analysis
    pitch_std = features["pitch_std"]
    if pitch_std < 15:
        ai_score += 3.0
        ai_indicators.append("Extremely consistent pitch")
    elif pitch_std < 30:
        ai_score += 1.5
        ai_indicators.append("Moderately consistent pitch")
    else:
        human_score += 2.0
        human_indicators.append("Natural pitch variation")

    if features["pitch_range"] < 40:
        ai_score += 2.0
        ai_indicators.append("Limited pitch range")
    else:
        human_score += 1.5
        human_indicators.append("Wide pitch range")

    # Energy analysis
    energy_var = features["energy_variance"]
    if energy_var < 0.0005:
        ai_score += 3.0
        ai_indicators.append("Uniform energy levels")
    elif energy_var < 0.002:
        ai_score += 1.0
        ai_indicators.append("Low energy variation")
    else:
        human_score += 2.0
        human_indicators.append("Natural energy fluctuations")

    # Spectral
    if features["centroid_std"] < 400:
        ai_score += 2.0
        ai_indicators.append("Consistent spectral centroid")
    else:
        human_score += 1.5

    if features["bandwidth_std"] < 300:
        ai_score += 1.5
        ai_indicators.append("Stable spectral bandwidth")
    else:
        human_score += 1.0

    # ZCR
    if features["zcr_std"] < 0.015:
        ai_score += 2.0
        ai_indicators.append("Regular zero-crossing pattern")
    else:
        human_score += 1.5
        human_indicators.append("Variable zero-crossing rate")

    # MFCC
    if features["mfcc_0_std"] < 8 and features["mfcc_1_std"] < 8:
        ai_score += 2.5
        ai_indicators.append("Uniform MFCC patterns")
    else:
        human_score += 1.5

    # Contrast
    if features["contrast_std"] < 2.0:
        ai_score += 1.5
        ai_indicators.append("Low spectral contrast variation")
    else:
        human_score += 1.0

    # Classification
    total = ai_score + human_score
    if total > 0:
        if ai_score > human_score:
            classification = "AI_GENERATED"
            confidence = min(0.99, ai_score / total)
        else:
            classification = "HUMAN"
            confidence = min(0.99, human_score / total)
    else:
        classification = "HUMAN"
        confidence = 0.5

    # Explanation
    indicators = ai_indicators if classification == "AI_GENERATED" else human_indicators
    if classification == "AI_GENERATED":
        explanation = (
            f"Analysis detected synthetic voice characteristics using {len(ai_indicators)} indicators. "
            f"Key findings: {'; '.join(ai_indicators[:3])}. "
            f"Pitch variation: {pitch_std:.1f} Hz (AI typically <20 Hz), "
            f"energy variance: {energy_var:.4f} (AI typically <0.001). "
            f"Consistent spectral patterns typical of text-to-speech synthesis detected. "
            f"Language: {language}. Confidence: {confidence*100:.1f}%."
        )
    else:
        explanation = (
            f"Analysis confirmed natural human speech using {len(human_indicators)} indicators. "
            f"Key findings: {'; '.join(human_indicators[:3])}. "
            f"Pitch variation: {pitch_std:.1f} Hz, "
            f"energy variance: {energy_var:.4f}. "
            f"Natural spectral dynamics consistent with human vocal production detected. "
            f"Language: {language}. Confidence: {confidence*100:.1f}%."
        )

    return classification, round(confidence, 3), explanation, ai_indicators, human_indicators


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "operational",
        "service": "AI Voice Detection API",
        "version": "2.0.0",
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_formats": ["mp3"],
        "endpoints": {
            "detection": "/api/detect",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat(),
        "message": "Ready for hackathon evaluation"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/detect")
async def detect_voice(
    request: Request,
    x_api_key: str = Header(None)
):
    """
    Main detection endpoint.
    Accepts ANY JSON body format and extracts audio from it.
    """
    start_time = time.time()

    # === AUTH ===
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # === PARSE BODY - accepts ANY format ===
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    logger.info(f"Received body keys: {list(body.keys())}")

    # === EXTRACT LANGUAGE ===
    language = body.get("language", "English")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language. Use: {SUPPORTED_LANGUAGES}")

    # === EXTRACT AUDIO BASE64 ===
    # Try ALL possible field names the hackathon might send
    audio_base64 = (
        body.get("audio_base64_format") or
        body.get("audio_base64") or
        body.get("audio") or
        body.get("audioBase64") or
        body.get("base64_audio") or
        body.get("file") or
        body.get("audio_data") or
        body.get("input_audio")
    )

    if not audio_base64:
        raise HTTPException(
            status_code=400,
            detail=f"No audio data found. Received keys: {list(body.keys())}. Expected one of: audio_base64_format, audio, audio_base64"
        )

    # === DECODE AUDIO ===
    try:
        # Remove data URI prefix if present (e.g., "data:audio/mp3;base64,...")
        if "," in audio_base64 and audio_base64.startswith("data:"):
            audio_base64 = audio_base64.split(",")[1]

        audio_data = base64.b64decode(audio_base64)
        logger.info(f"Audio decoded: {len(audio_data)} bytes, Language: {language}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}")

    # === EXTRACT FEATURES ===
    try:
        features = extract_features(audio_data)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    # === DETECT ===
    try:
        classification, confidence, explanation, ai_ind, human_ind = detect_ai_voice(features, language)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    processing_time = int((time.time() - start_time) * 1000)

    # === RESPONSE ===
    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation,
        "language_detected": language,
        "processing_time_ms": processing_time,
        "features_analyzed": {
            "pitch_variation_hz": round(features["pitch_std"], 2),
            "pitch_range_hz": round(features["pitch_range"], 2),
            "energy_variance": round(features["energy_variance"], 6),
            "spectral_centroid_std": round(features["centroid_std"], 2),
            "duration_seconds": round(features["duration"], 2),
            "zero_crossing_rate_std": round(features["zcr_std"], 4),
            "mfcc_variation": round(features["mfcc_0_std"], 2)
        },
        "model_version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
