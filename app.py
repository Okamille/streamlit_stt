from pathlib import Path

import librosa
import streamlit as st
import torch

from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model, Wav2Vec2ForCTC


@st.cache
def load_models():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return model, tokenizer


model, tokenizer = load_models()

st.title("STT with HuggingFace")

file = st.file_uploader("Upload audio file", type=None)
if file:
    filename = Path("audio_files") / file.name
    data, sample_rate = librosa.load(filename)
    st.write(f"Uploading data with sample rate : {sample_rate} Hz")
    st.audio(str(filename))
    data = [
        data[i * sample_rate * 5 : (i + 1) * sample_rate * 5]
        for i in range(int(len(data) / (sample_rate * 5)) + 1)
    ]

    input_values = tokenizer(data, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    transcription = " ".join(transcription)
    st.write(transcription)
