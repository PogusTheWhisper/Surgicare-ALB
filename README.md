# SurgiCare

<img src="app_logo.png" width="250">

SurgiCare is an AI-powered wound analysis system designed to assist healthcare professionals and patients in wound assessment and care management. The application combines computer vision and natural language processing to provide automated wound classification and personalized care recommendations.

## Features

### Core Functionality

- **Automated Wound Classification**: Identifies five types of wounds (Abrasions, Bruises, Burns, Cuts, Normal skin)
- **Multi-language Support**: Available in Thai and English
- **Interactive Chat Interface**: AI-powered conversational assistant for wound care guidance
- **Multiple Input Methods**: Upload images, capture photos, or select from sample images
- **Feature Extraction**: Detailed wound characteristic analysis using biomedical vision-language models

### Technical Components

- **ONNX-optimized Classification Model**: Fast inference with GPU acceleration support
- **BiomedVLP-BioViL-T Integration**: Advanced biomedical feature extraction
- **Streamlit Web Interface**: User-friendly web application
- **OpenAI-compatible API**: Integration with Typhoon AI models for conversational responses

## Architecture

### Model Pipeline

1. **Image Preprocessing**: Resize, crop, and normalize input images
2. **Wound Classification**: ONNX-based CNN model for wound type prediction
3. **Feature Extraction**: Biomedical vision-language model for detailed wound analysis
4. **Response Generation**: AI-powered care recommendations based on wound characteristics

### Key Components

- `Surgicare.py`: Main Streamlit application
- `utils/extract_wound_class.py`: Wound classification using cached ONNX model
- `utils/extract_wound_features.py`: Feature extraction using BiomedVLP-BioViL-T

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for accelerated inference)

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- streamlit
- torch
- torchvision
- transformers
- onnx
- onnxruntime
- pillow
- openai
- requests

### Configuration

Create `.streamlit/secrets.toml` with your API key:

```toml
TYPHOON_API_KEY = "your_api_key_here"
```

## Usage

### Running the Application

```bash
streamlit run Surgicare.py
```

### Using the Interface

1. **Image Input**: Choose from three options:

   - Upload an image file (JPG, JPEG, PNG)
   - Capture a photo using your camera
   - Select from provided sample images

2. **Analysis**: Click "Analyze Wound" to get:

   - Wound classification with confidence scores
   - Detailed feature analysis
   - AI-generated care recommendations

3. **Interactive Chat**: Ask follow-up questions about wound care in the chat interface

4. **Settings**: Adjust model parameters in the sidebar:
   - Language selection (Thai/English)
   - LLM model selection
   - Temperature and token limits

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
| ---- | -------- | -------- |
| 1    | 0.9291   | 0.9184   |
| 2    | 0.9488   | 0.9425   |
| 3    | 0.9567   | 0.9184   |
| 4    | 0.9449   | 0.9184   |
| 5    | 0.9685   | 0.9419   |

**Average Performance**:

- **Average Accuracy**: 0.9496 (94.96%)
- **Average F1-Score**: 0.9419 (94.19%)

### Model Details

- **Architecture**: Fine-tuned CNN with ONNX optimization
- **Training Data**: Open-source wound classification dataset
- **Inference**: GPU-accelerated with FP16 quantization support
- **Feature Model**: Microsoft BiomedVLP-BioViL-T for biomedical analysis

## Dataset

The model is trained on the wound classification dataset available at:
[Kaggle Wound Classification Dataset](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification)

Dataset includes:

- Abrasions: Surface-level skin injuries
- Bruises: Subcutaneous bleeding without open wounds
- Burns: Thermal or chemical skin damage
- Cuts: Linear wounds with clean edges
- Normal: Healthy skin samples

## API Integration

The application integrates with OpenAI-compatible APIs for conversational responses:

- **Supported Models**: Typhoon v2.1-12b-instruct, Typhoon v2-70b-instruct
- **Customizable Parameters**: Temperature, top-p, max tokens
- **Streaming Responses**: Real-time chat interface

## Demo and Resources

- **Live Demo**: [https://surgicare-alb-demo.streamlit.app/](https://surgicare-alb-demo.streamlit.app/)
- **Technical Article**: [SurgiCare: AI Builder Undone](https://medium.com/@naphatsorn.contact/surgicare-ai-builder-undone-676a0865f57c)

## File Structure

```
├── Surgicare.py                 # Main application
├── utils/
│   ├── extract_wound_class.py   # Wound classification model
│   └── extract_wound_features.py # Feature extraction model
├── .streamlit/
│   ├── config.toml              # Streamlit configuration
│   └── secrets.toml             # API keys and secrets
├── careful_this_contain_wound_image/ # Sample wound images
├── finetune/                    # Training and evaluation results
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

## Performance Optimization

### Model Optimization

- **ONNX Runtime**: Optimized inference with graph optimization
- **Dynamic Quantization**: FP16 quantization for CUDA devices
- **Memory Management**: Efficient buffer allocation and reuse
- **Multi-threading**: Optimized CPU utilization

### Caching Strategy

- **Model Caching**: Streamlit resource caching for model components
- **Session State**: Persistent chat history and user preferences
- **Image Processing**: Efficient PIL and tensor operations

## Limitations and Disclaimers

- **Medical Disclaimer**: This application provides preliminary wound assessment only
- **Professional Consultation**: Always consult licensed medical professionals for serious conditions
- **Accuracy**: Model performance may vary with image quality and lighting conditions
- **Scope**: Limited to five wound types in the training dataset

## Contributing

This project focuses on improving wound care accessibility through AI technology. Contributions should maintain the balance between technical accuracy and user-friendly interface design.
