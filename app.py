from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import pandas as pd
import shutil
import traceback
from src.feature_extraction import count_pauses, extract_text_features, semantic_coherence
from src.preprocessing import transcribe_audio, convert_to_wav
from src.modeling import run_modeling
import librosa
import numpy as np
import tempfile
from pathlib import Path

app = Flask(__name__)

# Use temporary directories for serverless environment
def get_temp_dirs():
    temp_base = tempfile.gettempdir()
    raw_dir = os.path.join(temp_base, 'raw')
    transcript_dir = os.path.join(temp_base, 'transcripts')
    processed_dir = os.path.join(temp_base, 'processed')
    
    for dir_path in [raw_dir, transcript_dir, processed_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return raw_dir, transcript_dir, processed_dir

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

        # Get temporary directories
        raw_dir, _, _ = get_temp_dirs()
        
        # Save uploaded audio to temp directory
        audio_path = os.path.join(raw_dir, audio_file.filename)
        audio_file.save(audio_path)

        # Transcribe the audio
        try:
            text = transcribe_audio(audio_path)
            
            # Clean up
            os.remove(audio_path)
            
            return jsonify({"transcript": text})
        except Exception as e:
            if os.path.exists(audio_path):
                os.remove(audio_path)
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

        # Get temporary directories
        raw_dir, transcript_dir, processed_dir = get_temp_dirs()
        
        # Save uploaded audio
        audio_path = os.path.join(raw_dir, audio_file.filename)
        audio_file.save(audio_path)

        try:
            # Validate audio file
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < 0.1:  # Less than 100ms is likely an error
                raise ValueError("Audio file appears to be empty or corrupted. Please check the file and try again.")

            # Save transcript
            text_path = os.path.join(transcript_dir, audio_file.filename.replace(".wav", ".txt"))
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

            # Clean up temporary files
            os.remove(audio_path)
            os.remove(text_path)

            return jsonify({
                "message": "Processing complete.",
                "data": result_dict
            })

        except Exception as e:
            # Clean up on error
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(text_path):
                os.remove(text_path)
            raise e

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# For local development only
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
