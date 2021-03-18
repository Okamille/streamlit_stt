import streamlit as st
from transformers import pipeline


text_to_translate = st.text_input("To translate : ")

if text_to_translate:
    translator = pipeline("translation_en_to_fr")

    translated_text = translator(text_to_translate)

    st.write(translated_text[0]["translation_text"])
