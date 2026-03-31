# 🖼️ AI vs Real Image Detection

A deep learning project that detects whether an image is **AI-Generated** or **Real** using MobileNetV2 transfer learning.

---

## 📌 Project Overview

This project uses a pretrained **MobileNetV2** model fine-tuned on a custom dataset of real and AI-generated images. The model outputs a confidence score indicating whether an image is real or AI-generated.

---

## 📂 Project Structure
```
AI-vs-Real-Image-Detection/
├── Traning.ipynb        # Model training pipeline
├── Testing.ipynb        # Model testing and prediction
├── .gitignore           # Ignores model and cache files
└── README.md            # Project documentation
```

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Dense(128, relu)
  - Dense(1, sigmoid)
- **Input Size:** 224 x 224 x 3
- **Output:** 0 = AI-Generated, 1 = Real

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Epochs | 10 |
| Batch Size | 32 |
| Image Size | 224x224 |

---

## 🚀 How to Run

> No installation needed! Just open in Google Colab.

### Training
1. Open `Traning.ipynb` in Google Colab
2. Mount your Google Drive
3. Set your dataset path:
```python
train_dir = "/content/drive/MyDrive/dataset_split/train"
val_dir   = "/content/drive/MyDrive/dataset_split/val"
```
4. Run all cells
5. Model saves to your Google Drive as `image_identifier_model.h5`

### Testing
1. Open `Testing.ipynb` in Google Colab
2. Mount your Google Drive
3. Load your saved model:
```python
model_path = "/content/drive/MyDrive/image_identifier_model.h5"
```
4. Run all cells
5. Enter image path when prompted

---

## 📦 Dataset Structure
```
dataset_split/
├── train/
│   ├── real/            # Real images
│   └── ai/              # AI-generated images
└── val/
    ├── real/            # Real images
    └── ai/              # AI-generated images
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Validation Accuracy | 97.55% |
| Validation Loss | 0.0923 |

> Update these values after training your model.

---

## 📌 Notes

- Model file (`.h5`) is stored on Google Drive and is **not included** in this repository
- Dataset is **not included** — bring your own real vs AI-generated image dataset
- All code runs on **Google Colab** — no local setup required
- Recommended to use **GPU runtime** in Colab for faster training:
```
Runtime → Change Runtime Type → GPU
```

---

## 🔮 Future Plans

- [ ] Deploy as a web app using Gradio
- [ ] Host model on Hugging Face
- [ ] Add support for batch image prediction
- [ ] Improve accuracy with fine-tuning

---

## 👤 Author

**Vasista Dhavala**  
GitHub: [@vasista-dhavala](https://github.com/vasista-dhavala)
