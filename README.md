# AI vs Real Image Detector

A web app that estimates whether an uploaded image is more likely to be a real photograph or AI-generated.

## Try it online

**[Open the live app →](https://ai-vs-real-image.streamlit.app/)**

Upload a JPG, JPEG, PNG, BMP, or WEBP image, then select **Analyse image**.

The app shows:

- Whether the image is likely a real photograph or AI-generated
- A confidence percentage
- Real-photo and AI-generated score breakdowns
- Image details such as resolution, file size, format, and analysis time

> Results are estimates, not proof of where an image came from. Compression, filters, screenshots, edits, and unfamiliar AI generators can affect predictions.

## How it works

The detector uses a MobileNetV2-based image-classification model.

1. The uploaded image is converted to RGB.
2. It is resized to 224 × 224 pixels.
3. The model calculates how likely it is to be a real photograph.
4. The result is displayed as either **Real photograph** or **AI-generated image**.

- Real-photo score above 50% → Real photograph
- Real-photo score at or below 50% → AI-generated image

The confidence percentage shows how strongly the model leans toward its result. It does not guarantee that the result is correct.

## Run locally

### Requirements

- Python 3.10 or newer
- `image_identifier_model.h5` in the project folder

### Installation

```bash
git clone https://github.com/vasista-dhavala/AI-vs-Real-Image-Detection.git
cd AI-vs-Real-Image-Detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt