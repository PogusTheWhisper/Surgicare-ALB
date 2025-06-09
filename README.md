# Surgicare

> **Surgicare (Surgical + Care)**

<img src="app_logo.png" width="250">

SurgiCare is an AI system designed to support post-surgery patient recovery. In this repository, we focus on a wound classification model trained on an open-source dataset. Our objective is to improve the accuracy of wound detection and guide patients in managing their wound recovery efficiently.

## Dataset

**Wound Dataset**: [https://www.kaggle.com/datasets/ibrahimfateen/wound-classification](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification)

## Demo and Documentation

- **Medium Article**: [SurgiCare: AI Builder Undone](https://medium.com/@naphatsorn.contact/surgicare-ai-builder-undone-676a0865f57c)
- **Live Demo**: [https://surgicare-alb-demo.streamlit.app/](https://surgicare-alb-demo.streamlit.app/)

## Model Performance

### Dataset Analysis
- **Dataset Distribution (Before)**: `finetune/dataset_before.png`
- **Dataset Distribution (After)**: `finetune/dataset_after.png`
- **Confusion Matrix**: `finetune/cfm.png`

### Classification Results

#### Overall Classification Report
```
              precision    recall  f1-score   support
   Abrasions       0.98      0.94      0.96        50
     Bruises       0.97      0.96      0.97        73
       Burns       0.95      0.98      0.96        41
         Cut       0.91      0.97      0.94        30
      Normal       1.00      1.00      1.00        60

    accuracy                           0.97       254
   macro avg       0.96      0.97      0.96       254
weighted avg       0.97      0.97      0.97       254
```

#### Class-wise Accuracy
- **Abrasions**: 94.00%
- **Bruises**: 95.89%
- **Burns**: 97.56%
- **Cut**: 96.67%
- **Normal**: 100.00%

### Cross-Validation Results

#### 5-Fold Cross-Validation Summary
| Fold | Accuracy | F1-Score |
|------|----------|----------|
| 1    | 0.9291   | 0.9184   |
| 2    | 0.9488   | 0.9425   |
| 3    | 0.9567   | 0.9184   |
| 4    | 0.9449   | 0.9184   |
| 5    | 0.9685   | 0.9419   |

**Average Performance**:
- **Average Accuracy**: 0.9496 (94.96%)
- **Average F1-Score**: 0.9419 (94.19%)