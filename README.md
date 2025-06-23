# Handwritten Digit Generator Web App

A web application that generates handwritten digits using a conditional GAN trained on the MNIST dataset. Users can select a digit (0-9) and the app will generate 5 sample images.

## Features
- Digit selection (0-9)
- Generates 5 unique handwritten digit images
- Responsive web interface
- Public deployment via Streamlit
- PyTorch implementation

## File Structure
handwritten-digit-generator/
├── .streamlit/
│ └── config.toml
├── app.py # Streamlit web application
├── cgan_mnist.py # GAN training script
├── generator.pth # Pre-trained generator weights
├── requirements.txt # Python dependencies
├── runtime.txt # Python version specification
└── README.md # This file


## Prerequisites
- Python 3.10
- Google Account (for Colab training)
- GitHub Account
- Streamlit Account

## Setup Instructions

### 1. Local Development

# Clone repository
git clone https://github.com/your-username/handwritten-digit-generator-by-mehtab-khan.git
cd handwritten-digit-generator-by-mehtab-khan

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

#### 2. Train the Model (Google Colab)
Open Google Colab

Upload cgan_mnist.py

Run with T4 GPU:

!pip install torch==2.3.0 torchvision==0.18.0
!python cgan_mnist.py

Download generator.pth after training completes

### 3. Run Streamlit App Locally

streamlit run app.py
Visit http://localhost:8501 in your browser

Deployment on Streamlit Cloud
Upload to GitHub:

Push all files to a new repository

Include generator.pth (use Git LFS if >100MB)

Create Streamlit App:

Sign in to Streamlit Community Cloud

Click "New app"

Connect your GitHub repository

Set configuration:

Repository: your-username/handwritten-digit-generator-by-mehtab-khan

Branch: main

Main file path: app.py

Advanced settings:

Python version: 3.10

Dependency management: Pip

Deploy:

Click "Deploy"

Wait 3-5 minutes for build process

App will be available at: https://handwritten-digit-generator-by-mehtab-khan.streamlit.app

Usage
Select a digit (0-9) from the dropdown

Click "Generate Digits"

View 5 generated handwritten digit images

(Optional) Upload new model weights using the file uploader

Troubleshooting
Common Issues
Error	Solution
ModuleNotFoundError	Run pip install -r requirements.txt
Model loading errors	Re-train model with compatible PyTorch version
Blank images	Check denormalization in app.py: (img + 1) / 2
Deployment failures	Set Python version to 3.10 in Streamlit settings
Streamlit Deployment Tips
Ensure runtime.txt contains 3.10

Verify generator.pth is <100MB (or use Git LFS)

Check build logs for dependency errors

If app crashes on startup, add this to app.py:

python
# Add to top of app.py
import os, sys
if sys.version_info >= (3, 12):
    os.system("pip install --force-reinstall -r requirements.txt")
Technical Specifications
Framework: PyTorch 2.3.0

Model: Conditional GAN

Dataset: MNIST (28x28 grayscale)

Training Hardware: Google Colab T4 GPU

Deployment: Streamlit Community Cloud

License
This project is licensed under the MIT License - see LICENSE file for details.

Maintained by Mehtab Khan
https://static.streamlit.io/badges/streamlit_badge_black_white.svg


## Key Components Explained

1. **Project Overview**: Clear description with visual placeholder
2. **File Structure**: Shows all necessary files and organization
3. **Setup Instructions**: 
   - Local development with virtual environment
   - Google Colab training with exact commands
4. **Deployment Guide**: Step-by-step Streamlit deployment
5. **Troubleshooting**: Common errors and solutions
6. **Technical Specs**: Hardware/software requirements
7. **License**: Default MIT license

## Additional Recommendations

1. Create a `.gitignore` file with:
venv/
pycache/
.ipynb_checkpoints/
data/
*.pyc


2. For large `generator.pth` files:
- Install Git LFS: `git lfs install`
- Track model: `git lfs track "*.pth"`
- Commit normally

3. Add a sample image to your repository:
- Create `/images/sample.png`
- Update README placeholder URL: `![Generated Digits Example](/images/sample.png)`

This README provides comprehensive instructions for all aspects of your project - from local setup to deployment troubleshooting. The format is professional yet accessible, with clear sections for different user needs.














