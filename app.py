import streamlit as st
from transformers import pipeline

st.title("AI Text Generation")
st.write("Generate text using DistilGPT2")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        result = generator(
            prompt,
            max_new_tokens=80,
            num_return_sequences=1
        )
        st.write(result[0]["generated_text"])
    else:
        st.warning("Please enter a prompt.")