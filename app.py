from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import shutil
import traceback
from src.feature_extraction import count_pauses, extract_text_features, semantic_coherence
from src.preprocessing import transcribe_audio, convert_to_wav
from src.modeling import run_modeling
import librosa
import numpy as np

app = Flask(__name__)

RAW_DIR = "data/raw"
TRANSCRIPT_DIR = "data/transcripts"
PROCESSED_DIR = "data/processed"
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "features_output.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio_file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio_file"]
        
        if not audio_file.filename:
            return jsonify({"error": "No selected file"}), 400

        if not audio_file.filename.endswith('.wav'):
            return jsonify({"error": "Only WAV files are supported"}), 400

        # Save uploaded audio
        audio_path = os.path.join(RAW_DIR, audio_file.filename)
        audio_file.save(audio_path)

        # Transcribe the audio
        try:
            text = transcribe_audio(audio_path)
            return jsonify({"transcript": text})
        except Exception as e:
            return jsonify({"error": f"Error transcribing audio: {str(e)}"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/process_audio", methods=["POST"])
def process_audio():
    try:
        if "audio_file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        if "transcript" not in request.form:
            return jsonify({"error": "No transcript provided"}), 400

        audio_file = request.files["audio_file"]
        transcript = request.form["transcript"]

        if not audio_file.filename:
            return jsonify({"error": "No selected file"}), 400

        if not audio_file.filename.endswith('.wav'):
            return jsonify({"error": "Only WAV files are supported"}), 400

        # Save uploaded audio
        audio_path = os.path.join(RAW_DIR, audio_file.filename)
        audio_file.save(audio_path)

        # Validate audio file
        try:
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < 0.1:  # Less than 100ms is likely an error
                return jsonify({"error": "Audio file appears to be empty or corrupted. Please check the file and try again."}), 400
        except Exception as e:
            return jsonify({"error": f"Error processing audio file: {str(e)}"}), 400

        # Save transcript
        text_path = os.path.join(TRANSCRIPT_DIR, audio_file.filename.replace(".wav", ".txt"))
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # Feature extraction
        pause_count, pause_avg = count_pauses(audio_path)
        hesitations, lexical_div, incomplete = extract_text_features(transcript)
        semantic_sim = semantic_coherence(transcript)
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
        
        # Convert numpy values to Python native types for JSON serialization
        result_dict = final_df.to_dict(orient="records")
        for record in result_dict:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value) if isinstance(value, np.floating) else int(value)

        return jsonify({
            "message": "Processing complete.",
            "data": result_dict
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
