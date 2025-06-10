# 🌾 Crop Recommender System

This project recommends the most suitable crops for cultivation based on soil nutrient levels, and environmental factors. It uses machine learning to assist farmers and agricultural experts in making informed decisions.

---

## 🚀 Features

- Predicts optimal crops using machine learning algorithms
- Inputs: Soil nutrient like pH, N, P, K etc.
- Web interface built with **Streamlit**
- PDF report generation for user inputs and prediction results


## 🧠 Tech Stack

- **Python 3.9+**
- **Pandas, NumPy, Scikit-learn, XGBoost**
- **Streamlit** – for web UI
- **ReportLab** – for PDF report generation
- **Matplotlib / Seaborn** – for data visualization

---

## 📊 How It Works

1. User enters soil nutrient data via Streamlit UI
2. Trained ML model predicts the most suitable crop
3. User can download a PDF report of their input and the prediction

---

## 📦 Installation

```bash
git clone https://github.com/code-with-prabhu/CropRecomendation.git
cd CropRecomendation
pip install -r requirements.txt
