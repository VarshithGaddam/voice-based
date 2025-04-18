# Voice Cognitive Detection

A Streamlit application that analyzes voice recordings to detect cognitive patterns and provide insights.

## Features

- Upload WAV audio files
- Automatic transcription
- Speech pattern analysis
- Cognitive metrics calculation
- Risk assessment
- Visual results presentation

## Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install ffmpeg:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Run the application:
   ```
   streamlit run streamlit_app.py
   ```

## Deployment on Streamlit Cloud

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Deploy the application

## Project Structure

- `streamlit_app.py`: Main Streamlit application
- `src/`: Source code for feature extraction, preprocessing, and modeling
- `.streamlit/`: Streamlit configuration files
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.12+
- FFmpeg
- See `requirements.txt` for Python packages
