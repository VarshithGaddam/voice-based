# run_pipeline.py

import os
import pandas as pd
from src.preprocessing import preprocess_audio_dir
from src.feature_extraction import count_pauses, extract_text_features, semantic_coherence
from src.modeling import run_modeling
from src.visualization import save_all_plots  # NEW
import librosa

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TRANSCRIPT_DIR = "data/transcripts"
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "features_output.csv")
PLOTS_DIR = "plots"

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    preprocess_audio_dir(RAW_DIR, PROCESSED_DIR, TRANSCRIPT_DIR)
    
    rows = []
    for fname in os.listdir(PROCESSED_DIR):
        if fname.endswith(".wav"):
            audio_path = os.path.join(PROCESSED_DIR, fname)
            text_path = os.path.join(TRANSCRIPT_DIR, fname.replace(".wav", ".txt"))

            with open(text_path) as f:
                transcript = f.read()

            pause_count, pause_avg = count_pauses(audio_path)
            hesitations, lexical_div, incomplete = extract_text_features(transcript)
            semantic_sim = semantic_coherence(transcript)
            speech_rate = len(transcript.split()) / (librosa.get_duration(path=audio_path) / 60)

            rows.append({
                "sample_id": fname,
                "pause_count": pause_count,
                "pause_avg_duration": pause_avg,
                "speech_rate": speech_rate,
                "pitch_variability": 0,  # Placeholder
                "hesitation_count": hesitations,
                "lexical_diversity": lexical_div,
                "incomplete_sentences": incomplete,
                "semantic_similarity": semantic_sim
            })

    df = pd.DataFrame(rows)
    final_df = run_modeling(df)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved features to {OUTPUT_CSV}")

    # Save visualizations
    save_all_plots(final_df, PLOTS_DIR)
    print(f"Saved visualizations to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
