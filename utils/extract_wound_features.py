from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import warnings

# Globals to store the CLIP model and processor
model = None
processor = None

def load_clip():
    """Lazy-load the CLIP model and processor with offline fallback."""
    global model, processor
    if model is None or processor is None:
        try:
            print("🔄 Trying to load CLIP model from Hugging Face...")
            # Add context manager to handle initialization error
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                # Add kwargs to avoid the init_empty_weights issue
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32", 
                    low_cpu_mem_usage=False,  # Avoid using init_empty_weights
                    _fast_init=False  # Skip some optimizations that might trigger the error
                )
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ Loaded CLIP model from Hugging Face.")
        except Exception as e:
            print(f"⚠️ Online model load failed: {e}")
            print("📦 Trying to load CLIP model from local fallback path...")
            try:
                local_model_path = "./models/clip/clip_model"
                local_processor_path = "./models/clip/clip_processor"

                if not os.path.isdir(local_model_path) or not os.path.isdir(local_processor_path):
                    raise FileNotFoundError("Local model or processor directory not found.")

                # Apply the same parameters to local loading
                model = CLIPModel.from_pretrained(
                    local_model_path,
                    low_cpu_mem_usage=False,
                    _fast_init=False
                )
                processor = CLIPProcessor.from_pretrained(local_processor_path)
                print("✅ Loaded CLIP model from local paths.")
            except Exception as e_local:
                print(f"❌ Failed to load CLIP model locally: {e_local}")
                model = None
                processor = None

def extract_wound_features(image_path):
    """
    Extract wound-related feature similarity scores from an image using CLIP.

    Args:
        image_path (str): Path to the wound image.

    Returns:
        dict: A dictionary of {feature_description: similarity_score}
    """
    # Ensure model is loaded
    load_clip()
    if model is None or processor is None:
        print("⚠️ Warning: CLIP model not available. Returning empty feature set.")
        return {}

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return {}

    # Encode image
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Feature descriptions
    features = [
        'Wound Color: Red', 'Wound Color: Purple', 'Wound Color: Yellow',
        'Wound Color: White', 'Wound Color: Black', 'Presence of Pus',
        'Presence of Scab', 'Wound Swelling', 'Wound Temperature: Warm',
        'Wound Temperature: Normal', 'Wound Odor: Unpleasant', 'Wound Odor: Neutral',
        'Wound Moisture: Dry', 'Wound Moisture: Moist', 'Wound Texture: Smooth',
        'Wound Texture: Rough', 'Pain Level: High', 'Pain Level: Low',
        'Wound Depth: Superficial', 'Wound Depth: Partial Thickness', 'Wound Depth: Full Thickness',
        'Wound Edges: Regular', 'Wound Edges: Irregular', 'Wound Edges: Undermined',
        'Skin Color: Normal', 'Skin Color: Hyperpigmented', 'Skin Color: Hypopigmented',
        'Skin Integrity: Intact', 'Skin Integrity: Fragile', 'Skin Integrity: Inflamed',
    ]

    # Compute similarities
    feature_scores = {}
    for desc in features:
        text_inputs = processor(text=desc, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        similarity = torch.matmul(image_features, text_features.T)
        feature_scores[desc] = similarity.item()

    return feature_scores