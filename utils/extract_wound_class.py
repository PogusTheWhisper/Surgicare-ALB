import os
import numpy as np
import requests
from tensorflow import keras
from tensorflow.keras.preprocessing import image

class WoundClassificationModel:
    model_paths = {}  # Dictionary to hold model names and their paths
    current_model_name = None  # Track the currently loaded model name
    models = {}  # Dictionary to hold loaded models

    @classmethod
    def list_available_models(cls):
        """List available models."""
        print("Available models:")
        for model_name in [
            'SurgiCare-V1-large-turbo',
            'SurgiCare-V1-large',
            'SurgiCare-V1-medium',
            'SurgiCare-V1-small'
        ]:
            print(f"- {model_name}")

    @classmethod
    def load_model(cls, model_name):
        if model_name in cls.models:
            cls.current_model_name = model_name
            print(f"Model '{model_name}' is already loaded.")
            return cls.models[model_name]

        model_urls = {
            'SurgiCare-V1-large-turbo': 'https://huggingface.co/PogusTheWhisper/SurgiCare/resolve/main/SurgiCare-V1-large-turbo.keras',
            'SurgiCare-V1-large': 'https://huggingface.co/PogusTheWhisper/SurgiCare/resolve/main/SurgiCare-V1-large.keras',
            'SurgiCare-V1-medium': 'https://huggingface.co/PogusTheWhisper/SurgiCare/resolve/main/SurgiCare-V1-medium.keras',
            'SurgiCare-V1-small': 'https://huggingface.co/PogusTheWhisper/SurgiCare/resolve/main/SurgiCare-V1-small.keras'
        }

        if model_name not in model_urls:
            raise ValueError(f"Model '{model_name}' is not recognized.")

        if model_name in cls.model_paths and os.path.exists(cls.model_paths[model_name]):
            model_path = cls.model_paths[model_name]
        else:
            model_path = cls.download_model(model_urls[model_name], model_name + '.keras')
            cls.model_paths[model_name] = model_path

        print(f"Loading model from {model_path}...")
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")

        cls.models[model_name] = model
        cls.current_model_name = model_name
        return model

    @staticmethod
    def download_model(url, model_filename):
        """Download the model from the specified URL."""
        model_path = os.path.join(os.getcwd(), model_filename)
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print(f"Model downloaded and saved to {model_path}.")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download model from {url}. Reason: {e}")
        return model_path

    @classmethod
    def extract_wound_class(cls, img_path):
        if cls.current_model_name is None:
            raise ValueError("No model is currently loaded. Please load a model first.")

        class_labels = {
            0: 'Abrasions',
            1: 'Bruises',
            2: 'Burns',
            3: 'Cut',
            4: 'Diabetic Wounds',
            5: 'Laseration',
            6: 'Normal',
            7: 'Pressure Wounds',
            8: 'Surgical Wounds',
            9: 'Venous Wounds'
        }

        model = cls.models[cls.current_model_name]

        for target_size in [(224, 224), (300, 300)]:
            try:
                img_array = cls.preprocess_image(img_path, target_size)
                predictions = model.predict(img_array)
                predicted_index = np.argmax(predictions)
                predicted_class = class_labels[predicted_index]
                return predicted_class, predictions
            except Exception as e:
                print(f"Error with size {target_size}: {e}")

        return None

    @staticmethod
    def preprocess_image(img_path, target_size):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


# Utility functions for single-use inference (alternative to class-based)

def download_model(model_url, model_path='temp_model.keras'):
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as file:
                file.write(response.content)
            print("Model downloaded and saved.")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")
    return model_path

def extract_wound_class(img_path, model_name):
    model_urls = {
        'SurgiCare-V1-fast-best': ('https://huggingface.co/PogusTheWhisper/SurgiCare/resolve/main/SurgiCare-V1-fast-best.keras', 224),
        'SurgiCare-V1-mini-best': ('https://huggingface.co/PogusTheWhisper/SurgiCare/resolve/main/SurgiCare-V1-mini-best-model.keras', 224),
        'SurgiCare-V1-best': ('https://huggingface.co/PogusTheWhisper/SurgiCare/resolve/main/SurgiCare-V1-best.keras', 300)
    }

    if model_name not in model_urls:
        raise ValueError(f"Model '{model_name}' is not recognized.")

    model_url, image_size = model_urls[model_name]

    class_labels = {
        0: 'Abrasions',
        1: 'Bruises',
        2: 'Burns',
        3: 'Cut',
        4: 'Diabetic Wounds',
        5: 'Laseration',
        6: 'Normal',
        7: 'Pressure Wounds',
        8: 'Surgical Wounds',
        9: 'Venous Wounds'
    }

    model_path = download_model(model_url, model_path=model_name + '.keras')

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")

    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]

    return predicted_class
