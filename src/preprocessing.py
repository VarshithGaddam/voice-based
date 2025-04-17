# src/preprocessing.py

import os
import torch
import whisper
import librosa
import soundfile as sf
from pydub import AudioSegment

MODEL = whisper.load_model("base")

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def transcribe_audio(audio_path):
    result = MODEL.transcribe(audio_path)
    return result['text']

def preprocess_audio_dir(raw_dir, processed_dir, transcript_dir):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):
        if filename.endswith((".wav", ".mp3")):
            input_path = os.path.join(raw_dir, filename)
            output_path = os.path.join(processed_dir, filename.replace(".mp3", ".wav"))
            
            # Convert to WAV & preprocess
            convert_to_wav(input_path, output_path)
            
            # Transcribe
            text = transcribe_audio(output_path)
            with open(os.path.join(transcript_dir, filename.replace(".wav", ".txt")), "w") as f:
                f.write(text)
