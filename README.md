# 🔬 Melanoma Skin Cancer Detection

A deep learning binary classification model that detects **melanoma skin cancer** from dermoscopy images using **EfficientNetB6** transfer learning with TensorFlow and Keras.

---

## 📌 Overview

This project trains a convolutional neural network to classify skin lesion images as:
- ✅ **Benign** — non-cancerous
- ❌ **Malignant** — cancerous (melanoma)

Transfer learning from a pre-trained **EfficientNetB6** (ImageNet) is used as the feature extractor, with a custom classification head on top.

---

## 🗂️ Project Structure

```
melanoma-detection/
│
├── melanoma_detection.ipynb   # Main Jupyter Notebook (Google Colab)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to ignore in Git
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## 🧠 Model Architecture

| Layer | Details |
|-------|---------|
| Input | 180 × 180 × 3 (RGB) |
| Data Augmentation | RandomFlip, RandomRotation, RandomZoom |
| Rescaling | Pixel values normalized to [0, 1] |
| EfficientNetB6 | Pre-trained on ImageNet, frozen |
| Flatten | Converts feature maps to 1D |
| Dense + Dropout | 512 → 0.5 drop → 256 → 0.3 drop → 128 |
| Output | Dense(1, sigmoid) — binary probability |

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Binary Cross-Entropy
- **Callbacks:** EarlyStopping (patience = 3)

---

## 📦 Dataset

The dataset is sourced from Kaggle:

🔗 [Melanoma Cancer Dataset — bhaveshmittal](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset)

It contains labeled images of benign and malignant skin lesions split into `train/` and `test/` directories.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/melanoma-detection.git
cd melanoma-detection
```

### 2. Open in Google Colab

Upload `melanoma_detection.ipynb` to [Google Colab](https://colab.research.google.com/) or click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 3. Set Up Kaggle API

- Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New API Token**
- Download `kaggle.json`
- Run the Kaggle setup cells in the notebook to upload it

### 4. Run All Cells

Run the notebook from top to bottom. The notebook will:
1. Download and extract the dataset
2. Build and train the model
3. Plot accuracy/loss curves
4. Predict on a test image
5. Save the model as `cancer_model.h5`

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~XX% |
| Validation Loss | ~X.XX |
| Epochs Trained | Up to 10 (EarlyStopping) |

> Update this table with your actual results after training.

---

## 🛠️ Requirements

See `requirements.txt`. Main dependencies:

- `tensorflow >= 2.10`
- `opencv-python`
- `numpy`
- `matplotlib`
- `kaggle`

---

## 💾 Saving & Loading the Model

The trained model is saved as:
```
cancer_model.h5
```

To reload it later:
```python
import tensorflow as tf
model = tf.keras.models.load_model('cancer_model.h5')
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset by [bhaveshmittal on Kaggle](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset)
- [EfficientNet paper](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- TensorFlow / Keras documentation
