
# 🧠 Racial Hoax Detection with Transformers and Custom Architectures

This project implements a multilingual racial hoax detection system using BERT, XLM-RoBERTa, CNN-on-BERT, and Transformer+FFNN models. It includes full training, evaluation, and comparison pipelines.

## 📁 Files

- `Detecting Racial Hoaxes.ipynb`: Main notebook with training, evaluation, and metrics visualization
- `model_comparison.csv`: CSV table with metrics from each model
- `model_comparison_report.txt`: Text report with detailed per-model performance
- `confusion_matrices.png`: Confusion matrices for all models
- `roc_curves.png`: ROC curves
- `precision_recall_curves.png`: Precision-recall curves
- `recall_comparison.png`: Bar plot comparing recall scores
- `model_comparison.png`: Combined metrics plot

## 🛠️ Features

- Multilingual tokenizer support (BERT/XLM-R)
- CNN and Transformer+FFNN built on top of frozen BERT
- Weighted loss to address class imbalance
- Model saving and prediction interface
- Visual comparison of classification metrics

## 📊 Models Compared

- `bert`: Multilingual BERT for sequence classification
- `xlm-roberta`: XLM-RoBERTa base
- `cnn`: CNN layer on top of frozen BERT outputs
- `transformer-ffnn`: Transformer encoder + FFNN on BERT embeddings

## 🚀 How to Run

1. Clone the repo or open in Colab
2. Place your train/val CSV files in the correct path (columns: `clean_text`, `labels`)
3. Run all cells in `Detecting Racial Hoaxes.ipynb`

## 🔍 Sample Prediction

```python
text = "ahamadtalwar ki nok par tumahre amao ne saya khol diya tha"
prediction, prob = detector.predict(text)
```

## 📦 Dependencies

- `transformers`
- `torch`
- `pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `tqdm`



