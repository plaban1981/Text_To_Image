import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import time

@st.experimental_singleton
def init_diffusers():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return pipe.to(device)

diffusers = init_diffusers()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device).manual_seed(1024)
my_bar = st.progress(0)

# Your long-running model
def run_model():
    for i in range(100):
        time.sleep(0.1)
        yield i

st.write("""
## ⚡️ Image to Text ⚡️
""")

# Take a text input
prompt = st.text_input("Enter your text:")


if prompt != "":
    with st.spinner(text="Generating image...It might take a while..."):
        image = diffusers(prompt,num_inference_steps=75,generator=generator).images[0]
          

    with st.spinner(text="Image Generated 🚀🚀🚀"):
        st.image(image, use_column_width=True)
