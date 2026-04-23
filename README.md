# 📄 Document Denoising using Attention U-Net

This project focuses on removing noise from grayscale document images using deep learning. It includes model training (via notebook) and a deployed inference pipeline using a web API and browser-based interface.

---

## 🚀 Overview

* **Goal:** Restore noisy document scans into clean, readable images
* **Model:** Attention U-Net (encoder-decoder architecture)
* **Input:** Noisy grayscale images
* **Output:** Denoised grayscale images

---

## 🧠 Key Features

* Deep learning-based document enhancement
* Attention U-Net for improved feature selection
* End-to-end pipeline: training → inference → web interface
* FastAPI backend for real-time denoising
* Browser UI with before/after comparison

---

## 📁 Project Structure

```text
Document-Denoising/
├── app.py                  # FastAPI inference server
├── best_model_v2.pth       # Trained model weights
├── extracted_code.py       # Standalone inference and model definitions
├── pipline.ipynb           # Training notebook
├── requirements.txt
├── README.md
├── static/
│   └── index.html          # Frontend UI
└── data/
    ├── train/
    └── train_cleaned/
```

---

## 📊 Dataset

* **Type:** Paired dataset (noisy → clean)
* **Size:** 144 image pairs
* **Format:** Grayscale images

```
data/train/           → noisy images  
data/train_cleaned/   → clean images  
```

---

## ⚙️ Training Details

Training is done inside `pipline.ipynb`.

### Preprocessing

* Convert to grayscale
* Resize to **256 × 256**
* Convert to tensor

### Models Used

* Baseline U-Net
* U-Net with combined loss (`MSE + L1`)
* **Attention U-Net (final model)**

![Attention U-Net Architecture](architecture.png)

### Loss Function

```text
Loss = MSE + 0.5 × L1
```

### Training Config

* Epochs: 10
* Train/Validation split: 80/20
* Best model saved as: `best_model_v2.pth`

---

## 🧪 Run Training

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open:

```
pipline.ipynb
```

---

## 🌐 Inference (Web App)

The project includes a real-time inference pipeline using **FastAPI**.

### Run the server

```bash
uvicorn app:app --reload
```

### Open in browser

```
http://localhost:8000
```

---

## 🖥️ Web Interface Features

* Upload document image
* View **original vs denoised** side-by-side
* Maintains aspect ratio (no pixel distortion)
* Real-time processing

---

## 🔁 Inference Pipeline

![Inference Pipeline Flowchart](pipeline.png)

1. Upload image via browser
2. Convert to grayscale
3. Resize to 256×256
4. Run model inference
5. Clamp output to valid range
6. Resize back to original resolution
7. Return denoised image

---

## ⚠️ Important Notes

* Model was trained on **256×256 images** → inference must match
* Output is a **denoised image (not segmentation)**
* No thresholding or binarization should be applied
* Use high-quality resizing (`LANCZOS`) for best results

---

## 📈 Future Improvements

* Train on higher resolution images (512×512 or full-size)
* Add PSNR / SSIM evaluation metrics
* Integrate OCR (text extraction from cleaned documents)
* Batch processing for multiple files
* Deploy API (Render / AWS / Docker)
* Add before/after slider UI

---

## 🧠 Learning Outcomes

This project demonstrates:

* CNN-based image restoration
* Attention mechanisms in U-Net
* ML pipeline debugging (train vs inference mismatch)
* Backend deployment with FastAPI
* Full-stack AI application development

---

## 📌 Conclusion

This is a complete **AI-powered document enhancement system**, combining:

* Deep learning
* Backend engineering
* Frontend visualization

---

## 👨‍💻 Author

Mohsin Khan
Backend Developer | AI Enthusiast

---

## ⭐ If you found this useful

Consider improving it further or turning it into a production-ready tool 🚀
