# src/feature_extraction.py

import librosa
import numpy as np
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def count_pauses(audio_path, threshold_db=-30):
    y, sr = librosa.load(audio_path)
    intervals = librosa.effects.split(y, top_db=abs(threshold_db))
    total_speech = sum([(end - start) for start, end in intervals])
    pause_count = len(y) - total_speech
    pause_durations = [(end - start)/sr for start, end in intervals]
    return len(pause_durations), np.mean(pause_durations)

def extract_text_features(text):
    sents = sent_tokenize(text)
    words = word_tokenize(text)
    hesitation_count = len(re.findall(r'\b(uh+|um+)\b', text.lower()))
    lexical_diversity = len(set(words)) / len(words) if words else 0
    incomplete_sentences = sum(1 for sent in sents if not sent.endswith(('.', '?')))
    return hesitation_count, lexical_diversity, incomplete_sentences

def semantic_coherence(text):
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)
    similarities = []
    for i in range(1, len(embeddings)):
        similarities.append(cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0])
    return np.mean(similarities) if similarities else 0
