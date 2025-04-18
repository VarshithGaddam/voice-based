import streamlit as st
import os
import pandas as pd
import numpy as np
import tempfile
import librosa
from src.feature_extraction import count_pauses, extract_text_features, semantic_coherence
from src.preprocessing import transcribe_audio
from src.modeling import run_modeling

# Set page config
st.set_page_config(
    page_title="Voice Cognitive Detection",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Voice Cognitive Detection")
st.markdown("""
This application analyzes voice recordings to detect cognitive patterns and provide insights.
Upload a WAV file and get detailed analysis of speech patterns, hesitations, and more.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name
    
    # Transcribe the audio
    with st.spinner("Transcribing audio..."):
        try:
            transcript = transcribe_audio(audio_path)
            st.text_area("Transcript", transcript, height=150)
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            transcript = None
    
    # Clean up the temporary file
    os.unlink(audio_path)
    
    if transcript:
        # Analyze button
        if st.button("Analyze Audio"):
            with st.spinner("Analyzing audio..."):
                try:
                    # Validate audio file
                    y, sr = librosa.load(audio_path)
                    duration = librosa.get_duration(y=y, sr=sr)
                    if duration < 0.1:  # Less than 100ms is likely an error
                        st.error("Audio file appears to be empty or corrupted. Please check the file and try again.")
                    else:
                        # Feature extraction
                        pause_count, pause_avg = count_pauses(audio_path)
                        hesitations, lexical_div, incomplete = extract_text_features(transcript)
                        semantic_sim = semantic_coherence(transcript)
                        speech_rate = len(transcript.split()) / (duration / 60)

                        # Create feature dataframe
                        rows = [{
                            "sample_id": uploaded_file.name,
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
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Convert to a more readable format
                        result_dict = final_df.to_dict(orient="records")[0]
                        
                        # Create two columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Speech Metrics")
                            st.metric("Pause Count", f"{result_dict['pause_count']:.2f}")
                            st.metric("Average Pause Duration", f"{result_dict['pause_avg_duration']:.2f} seconds")
                            st.metric("Speech Rate", f"{result_dict['speech_rate']:.2f} words/minute")
                            st.metric("Hesitation Count", f"{result_dict['hesitation_count']:.2f}")
                        
                        with col2:
                            st.markdown("### Cognitive Metrics")
                            st.metric("Lexical Diversity", f"{result_dict['lexical_diversity']:.2f}")
                            st.metric("Incomplete Sentences", f"{result_dict['incomplete_sentences']:.2f}")
                            st.metric("Semantic Similarity", f"{result_dict['semantic_similarity']:.2f}")
                            st.metric("Risk Score", f"{result_dict['risk_score']:.2f}")
                        
                        # Display cluster and anomaly information
                        st.markdown("### Classification")
                        cluster_status = "Normal" if result_dict['cluster'] == 0 else "Abnormal"
                        anomaly_status = "Normal" if result_dict['anomaly'] == 0 else "Anomaly"
                        
                        st.info(f"**Cluster Classification:** {cluster_status}")
                        st.info(f"**Anomaly Detection:** {anomaly_status}")
                        
                        # Add interpretation
                        st.markdown("### Interpretation")
                        if result_dict['risk_score'] < 0.3:
                            st.success("The speech pattern appears to be within normal cognitive parameters.")
                        elif result_dict['risk_score'] < 0.6:
                            st.warning("The speech pattern shows some signs of cognitive changes. Consider further evaluation.")
                        else:
                            st.error("The speech pattern shows significant signs of cognitive changes. Professional evaluation recommended.")
                
                except Exception as e:
                    st.error(f"Error analyzing audio: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Voice Cognitive Detection Tool | Developed for cognitive assessment") 