# app.py
import streamlit as st
import torch
import torch.nn as nn  # ADDED MISSING IMPORT
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import subprocess
import sys

# Install required packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install PyTorch if missing
try:
    import torch
    import torch.nn as nn
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("Installing PyTorch...")
    install("torch==2.3.0")
    install("torchvision==0.18.0")
    import torch
    import torch.nn as nn

# Generator Model Definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, 28, 28)

# Load pre-trained generator
@st.cache_resource
def load_model():
    generator = Generator()
    generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    generator.eval()
    return generator

# Generate images
def generate_digits(model, digit, num_images=5):
    z = torch.randn(num_images, 100)
    labels = torch.tensor([digit] * num_images)
    with torch.no_grad():
        generated = model(z, labels).detach().cpu()
    return generated

# Streamlit UI
st.title('Handwritten Digit Generator by MEHTAB KHAN')
digit = st.selectbox('Select digit to generate', options=list(range(10)), index=0)

if st.button('Generate Digits'):
    model = load_model()
    generated = generate_digits(model, digit)
    
    # Display images in a grid
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        img = generated[i].squeeze().numpy()
        img = (img + 1) / 2  # Denormalize
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Digit {digit}')
    st.pyplot(fig)

st.markdown("""
### About
- Generated using conditional GAN trained on MNIST
- Model trained from scratch with PyTorch
""")
