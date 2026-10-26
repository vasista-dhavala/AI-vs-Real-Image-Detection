import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / "image_identifier_model.h5"
IMG_SIZE = (224, 224)
PREVIEW_SIZE = (320, 320)


def build_model():
    """Rebuild the training architecture and load the original trained weights."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    base.trainable = False
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.load_weights(str(MODEL_PATH))
    return model


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Generated Image Detector")
        self.root.geometry("460x620")
        self.root.resizable(False, False)

        self.model = None
        self.image_path = None
        self.preview_photo = None

        tk.Label(root, text="AI vs Real Image Detector",
                 font=("Helvetica", 18, "bold")).pack(pady=(15, 5))
        tk.Label(root, text="Fine-tuned MobileNetV2",
                 font=("Helvetica", 11)).pack()

        self.preview = tk.Label(root, text="No image selected", width=40, height=15,
                                relief="groove", bg="#eeeeee", fg="#555555")
        self.preview.pack(pady=15)

        btn_frame = tk.Frame(root)
        btn_frame.pack()
        self.select_btn = tk.Button(btn_frame, text="Select Image", width=14,
                                    command=self.select_image)
        self.select_btn.grid(row=0, column=0, padx=8)
        self.classify_btn = tk.Button(btn_frame, text="Classify Image", width=14,
                                      command=self.classify, state="disabled")
        self.classify_btn.grid(row=0, column=1, padx=8)

        self.result_label = tk.Label(root, text="Result: -", font=("Helvetica", 16, "bold"))
        self.result_label.pack(pady=(25, 5))
        self.conf_label = tk.Label(root, text="Confidence: -", font=("Helvetica", 13))
        self.conf_label.pack()

        self.status = tk.Label(root, text="Loading model...", fg="#555555")
        self.status.pack(side="bottom", pady=10)

        self.root.after(100, self.load_model)

    def load_model(self):
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")
            self.model = build_model()
            self.status.config(text="Model loaded. Ready.", fg="#2e7d32")
            self.update_classify_state()
        except Exception as e:
            self.status.config(text="Model could not be loaded.", fg="#c62828")
            messagebox.showerror("Model error", str(e))

    def update_classify_state(self):
        ok = self.model is not None and self.image_path is not None
        self.classify_btn.config(state="normal" if ok else "disabled")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            thumb = img.copy()
            thumb.thumbnail(PREVIEW_SIZE)
            self.preview_photo = ImageTk.PhotoImage(thumb)
            self.preview.config(image=self.preview_photo, text="", width=0, height=0)
            self.image_path = path
            self.result_label.config(text="Result: -", fg="black")
            self.conf_label.config(text="Confidence: -")
            self.status.config(text=Path(path).name, fg="#555555")
            self.update_classify_state()
        except Exception as e:
            messagebox.showerror("Image error", f"Could not open image:\n{e}")

    def classify(self):
        try:
            img = Image.open(self.image_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            arr = np.array(img, dtype="float32") / 255.0
            pred = float(self.model.predict(arr[np.newaxis], verbose=0)[0][0])

            # Training labels: real = 1, ai = 0
            if pred > 0.5:
                label, conf, color = "Real Image", pred * 100, "#2e7d32"
            else:
                label, conf, color = "AI-Generated Image", (1 - pred) * 100, "#c62828"

            self.result_label.config(text=f"Result: {label}", fg=color)
            self.conf_label.config(text=f"Confidence: {conf:.2f}%")
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()