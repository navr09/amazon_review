# Amazon Review Helpfulness Predictor

![GitHub](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![ML](https://img.shields.io/badge/ml-xgboost-orange)

A machine learning system to predict the helpfulness of Amazon product reviews using XGBoost, achieving **76.4% ROC-AUC** and **83.3% PR-AUC** performance.

---

## 📊 Key Metrics
| Metric               | Score      |
|----------------------|------------|
| **ROC-AUC**          | 0.764      |
| **PR-AUC**           | 0.833      |
| **Optimal Threshold**| 0.4        |
| **Accuracy**         | 0.72       |
| **F1-Score (Helpful)** | 0.81     |

### Classification Report
| Class         | Precision | Recall | Support  |
|---------------|-----------|--------|----------|
| **Helpful**   | 0.72      | 0.92   | 119,438  |
| **Unhelpful** | 0.71      | 0.40   | 68,022   |

---

## 🔍 Project Highlights

### 🎯 Key Predictors (SHAP Analysis)
1. **Helpfulness Signal (TF-IDF)**: Strongest predictor
2. **5-Star Ratings**: Positive correlation with helpfulness
3. **Verified Purchases**: Slight helpfulness boost
4. **Moderate-Length Reviews**: Most helpful (extremes less effective)

### 📈 Dataset Characteristics
- **1.2 million reviews** (2003-2005)
- **Books category**: 370,978 unique products
- **Helpfulness Distribution**:
  - Helpful (≥0.7 ratio): 63.9%
  - Unhelpful: 36.1%

---

## 🛠 Implementation

### 🔧 Preprocessing
- Removed HTML/special characters
- Deduplicated reviews (>3 occurrences)
- Filtered reviews with <3 votes
- Created binary target (helpful if ratio ≥0.7)

### 🛠️ Feature Engineering
| Feature Type          | Examples                          |
|-----------------------|-----------------------------------|
| **Text Metrics**      | Title/body length, word counts    |
| **Sentiment**         | Polarity, subjectivity            |
| **Enhanced Features** | TF-IDF helpfulness signal         |

### 🧠 Models Evaluated
1. **XGBoost** (Champion)
   - Regularization prevents overfitting
   - Handles missing values
2. LightGBM
3. Gradient Boosting

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/navr09/amazon_review.git
cd amazon_review
pip install -r requirements.txt

To run python3.11 amazon_usecase/main.py