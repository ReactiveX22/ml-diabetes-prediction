# Diabetes Prediction using Machine Learning

## Project Overview

This project develops and compares multiple machine learning models to predict diabetes risk based on clinical and demographic features. Three models were implemented and tuned: Logistic Regression, Random Forest, and XGBoost.

## Team Members

- 2104010202279 - Sanjida Mahmud Muntaha
- 2104010202291 - Tainur Rahaman
- 2104010202299 - Akibul Islam

**Section**: D  
**Batch**: 40th

## Dataset

- **Source**: [DiaBD.csv](https://data.mendeley.com/datasets/m8cgwxs9s6/3)
- **Samples**: 6,610 clinical records
- **Features**: 14 clinical/demographic features
- **Target**: Diabetic (Yes/No)

## Quick Start

To reproduce results and run the demo:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run inference demo**:
   ```bash
   python inference.py
   ```
   This will load all models and show performance on test data.

## File Structure

```
inference.py              # Demo script to test all models
ML_Project_v0_3.ipynb     # Complete notebook with all code
data/                     # Dataset and processed test files
models/                   # Saved models and preprocessing objects
results/                  # Saved results
```

## Models Implemented

- Logistic Regression (with L1/L2 regularization)
- Random Forest (200 estimators, balanced classes)
- XGBoost (GPU-accelerated, scale_pos_weight for imbalance)

## Results

- **Best model**: Random Forest with AUC-ROC: 0.8542

## Reproduction Instructions

- Run `inference.py` for quick demo
- Open `ML_Project_Final.ipynb` for full analysis
- All random seeds set to 42 for reproducibility
