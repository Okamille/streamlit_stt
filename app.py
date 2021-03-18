from pathlib import Path

import librosa
import streamlit as st
import torch

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC


@st.cache
def load_models():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53-french"
    )
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
    return model, tokenizer


model, tokenizer = load_models()

st.title("STT with HuggingFace")

file = st.file_uploader("Upload audio file", type=["wav"])
if file:
    data, sample_rate = librosa.load(file)
    st.write(f"Uploading data with sample rate : {sample_rate} Hz")
    st.audio(file.getvalue())
    input_values = tokenizer(data, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    st.write(transcription)
