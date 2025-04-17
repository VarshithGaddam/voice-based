from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import shutil
import traceback
from src.feature_extraction import count_pauses, extract_text_features, semantic_coherence
from src.modeling import run_modeling
import librosa

app = FastAPI()

RAW_DIR = "data/raw"
TRANSCRIPT_DIR = "data/transcripts"
PROCESSED_DIR = "data/processed"
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "features_output.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/process_audio/")
async def process_audio(audio_file: UploadFile = File(...), transcript: str = Form(...)):
    try:
        # Save uploaded audio
        audio_path = os.path.join(RAW_DIR, audio_file.filename)
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Save transcript
        text_path = os.path.join(TRANSCRIPT_DIR, audio_file.filename.replace(".wav", ".txt"))
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # Feature extraction
        pause_count, pause_avg = count_pauses(audio_path)
        hesitations, lexical_div, incomplete = extract_text_features(transcript)
        semantic_sim = semantic_coherence(transcript)
        duration = librosa.get_duration(filename=audio_path)
        if duration == 0:
            raise ValueError("Audio file has zero duration.")

        speech_rate = len(transcript.split()) / (duration / 60)

        rows = [{
            "sample_id": audio_file.filename,
            "pause_count": pause_count,
            "pause_avg_duration": pause_avg,
            "speech_rate": speech_rate,
            "pitch_variability": 0,  # Placeholder
            "hesitation_count": hesitations,
            "lexical_diversity": lexical_div,
            "incomplete_sentences": incomplete,
            "semantic_similarity": semantic_sim
        }]

        df = pd.DataFrame(rows)
        final_df = run_modeling(df)
        final_df.to_csv(OUTPUT_CSV, index=False)

        return JSONResponse({
            "message": "Processing complete.",
            "data": final_df.to_dict(orient="records")
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
