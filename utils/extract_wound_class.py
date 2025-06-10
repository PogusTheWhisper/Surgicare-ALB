import os
import requests
import tempfile
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
class CachedWoundClassifier:
    MODEL_URL = "https://huggingface.co/PogusTheWhisper/Surgicare-ALB-fold4-stage3/resolve/main/topdown_model_fold4_stage3_opset_20.onnx"
    MODEL_PATH = os.path.join(tempfile.gettempdir(), "topdown_model_fold4_stage3_opset_20.onnx")
    CLASS_LABELS = {
        0: 'Abrasions',
        1: 'Bruises',
        2: 'Burns',
        3: 'Cut',
        4: 'Normal'
    }
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if ort.get_available_providers().__contains__('CUDAExecutionProvider') else "cpu")
        self.model = self._load_or_download_model()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_or_download_model(self):
        if not os.path.exists(self.MODEL_PATH):
            print(f"Model not found. Downloading from: {self.MODEL_URL}")
            try:
                response = requests.get(self.MODEL_URL)
                response.raise_for_status()
                with open(self.MODEL_PATH, "wb") as f:
                    f.write(response.content)
                print("Model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        
        print("Loading ONNX model from local file...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        session = ort.InferenceSession(self.MODEL_PATH, providers=providers)
        print("ONNX model loaded and ready.")
        return session

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        return image.numpy()

    def predict(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        
        input_name = self.model.get_inputs()[0].name
        
        logits = self.model.run(None, {input_name: image_tensor})[0]
        
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        pred_idx = np.argmax(probs, axis=1).item()
        pred_label = self.CLASS_LABELS.get(pred_idx, "Unknown")
        
        return pred_label, probs