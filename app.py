import streamlit as st

from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
import soundfile as sf

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

st.title("STT with HuggingFace")

file = st.file_uploader("Upload audio file", type="wav")
data = sf.read(file)
print(data)
input_values = tokenizer(data, return_tensors="pt", padding="longest").input_values  # Batch size 1
