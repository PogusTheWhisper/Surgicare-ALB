write me README in export README.md with out emoji

"""

# Surgicare

> **Surgicare (Surgical + Care)**  
> <img src="app_logo.png" width="250">

SurgiCare is an AI system designed to support post-surgery patient recovery. In this repository, we focus on a wound classification model trained on an open-source dataset. Our objective is to improve the accuracy of wound detection and guide patients in managing their wound recovery efficiently.

- **Wound Dataset**: [https://www.kaggle.com/datasets/ibrahimfateen/wound-classification](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification)
---
"""

medium log is here: https://medium.com/@naphatsorn.contact/surgicare-ai-builder-undone-676a0865f57c
demo is here: https://surgicare-alb-demo.streamlit.app/

dataset plot before: finetune/dataset_before.png
dataset plot after: finetune/dataset_after.png
confuse matrix: finetune/cfm.png

Classification Report:
              precision    recall  f1-score   support

   Abrasions       1.00      0.88      0.93         8
     Bruises       1.00      1.00      1.00        12
       Burns       1.00      1.00      1.00         6
         Cut       0.83      1.00      0.91         5
      Normal       1.00      1.00      1.00        10

    accuracy                           0.98        41
   macro avg       0.97      0.97      0.97        41
weighted avg       0.98      0.98      0.98        41

Accuracy for class 'Abrasions': 87.50%
Accuracy for class 'Bruises': 100.00%
Accuracy for class 'Burns': 100.00%
Accuracy for class 'Cut': 100.00%
Accuracy for class 'Normal': 100.00%
Fold 2 Accuracy: 0.9756

Summary:
Fold 1 - Accuracy: 0.9468, F1: 0.9426
Fold 2 - Accuracy: 0.9681, F1: 0.9530
Fold 3 - Accuracy: 0.9362, F1: 0.9252
Fold 4 - Accuracy: 0.9149, F1: 0.8954
Fold 5 - Accuracy: 0.9149, F1: 0.8948

Avg Accuracy: 0.9362 | Avg F1: 0.9222
CPU times: user 33min 12s, sys: 1min 13s, total: 34min 26s
Wall time: 27min 6s