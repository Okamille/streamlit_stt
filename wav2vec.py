#%%
import librosa
import numpy
import torch
from transformers import (
    Wav2Vec2Tokenizer,
    Wav2Vec2ForCTC,
)


def sample_data(data: numpy.ndarray):
    sample = data


audio_file = "audio_files/english_audio.mp3"

data, sample_rate = librosa.load(audio_file)
data = [
    data[i * sample_rate * 5 : (i + 1) * sample_rate * 5]
    for i in range(int(len(data) / (sample_rate * 5)))
]

print(f"Sampling rate : {sample_rate} Hz")

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")

input_values = tokenizer(data, return_tensors="pt", padding="longest").input_values
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)
