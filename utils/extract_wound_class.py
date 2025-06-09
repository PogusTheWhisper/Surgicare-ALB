import os
import torch
import requests
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from functools import lru_cache


# ---- Custom Model Class ----
class WoundClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4):
        super().__init__()
        base = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        n_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base

        self.shared_head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )

        self.class_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.layer_groups = [
            self.backbone.features[0:2],
            self.backbone.features[2:4],
            self.backbone.features[4:6],
            self.backbone.features[6:]
        ]

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_head(x)
        cls_out = self.class_head(x)
        return cls_out


# ---- Classifier Handler ----
class CachedWoundClassifier:
    MODEL_URL = "https://huggingface.co/PogusTheWhisper/Surgicare-ALB-fold2-stage3/resolve/main/topdown_model_fold2_stage3.pt"
    MODEL_PATH = "topdown_model_fold2_stage3.pt"
    CLASS_LABELS = {
        0: 'Abrasions',
        1: 'Bruises',
        2: 'Burns',
        3: 'Cut',
        4: 'Normal'
    }

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model().to(self.device)
        self.model.eval()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        if not os.path.exists(CachedWoundClassifier.MODEL_PATH):
            print(f"Downloading model from {CachedWoundClassifier.MODEL_URL}...")
            response = requests.get(CachedWoundClassifier.MODEL_URL)
            response.raise_for_status()
            with open(CachedWoundClassifier.MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded.")

        print("Loading model state_dict into architecture...")
        model = WoundClassifier(num_classes=5)
        state_dict = torch.load(CachedWoundClassifier.MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
        return model

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_transform():
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image_path):
        transform = self._get_transform()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image.to(self.device)

    def predict(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = self.CLASS_LABELS.get(pred_idx, "Unknown")
            return pred_label, probs.cpu().numpy()
