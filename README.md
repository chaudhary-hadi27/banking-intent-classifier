# 🏦 Banking Intent Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/🎈-Streamlit-red)

Created by: **Chaudhary Hadi**

## 📋 Project Overview
Fine-tuned BERT model to classify banking customer queries into 77 different intents using the Banking77 dataset.

## 🎯 Features
- Classifies customer queries into 77 banking intents
- 93% accuracy on test set
- Confidence threshold for human escalation
- Interactive Streamlit web app

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-streamlit-url)

## 📊 Dataset
- **Banking77** dataset from Hugging Face
- 77 intent classes
- 10,003 training samples
- 3,080 test samples

## 🏗️ Model Architecture
- Base model: `bert-base-uncased`
- Fine-tuned for sequence classification
- 77 output classes

## 📈 Performance
| Metric | Score |
|--------|-------|
| Accuracy | 93% |
| Precision | 93.3% |
| Recall | 92.9% |
| F1-Score | 93.0% |

## 🛠️ Installation

```bash
git clone https://github.com/chaudhary-hadi27/banking-intent-classifier
cd banking-intent-classifier
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Dependencies
```
transformers

torch

streamlit

datasets

huggingface-hub
```

## 👨‍💻 Author
```
Chaudhary Hadi

GitHub: chaudhary-hadi27

X: @ChaudharyHadi27
```

## 🙏 Acknowledgments
```
Hugging Face for transformers library

Banking77 dataset creators
```

## 📄 License
* MIT License
---

## 🎯 **Final Recommendation:**

**GitHub Repo Name:** `banking-intent-classifier`  
**Streamlit App Title:** "Banking Intent Classifier by Chaudhary Hadi"

Simple, professional, and easy to remember! 🚀