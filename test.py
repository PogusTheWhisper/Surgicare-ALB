import gradio as gr
import spaces
import torch
import os
import gc
import warnings
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import (
    StableDiffusionXLInstructPix2PixPipeline,
    AutoencoderKL,
    EDMEulerScheduler,
)
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========== Device Setup ==========
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
elif device == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# ========== Memory Management ==========
def torch_gc():
    try:
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except:
        pass
    gc.collect()
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

# ========== Image Preprocessing ==========
def resize_and_pad(image, size=488):
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    delta_w, delta_h = size - new_w, size - new_h
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    padded = ImageOps.expand(resized, padding, fill=(255, 255, 255))
    return padded, (w, h), padding

def crop_and_upscale(image, original_size, padding):
    left, top, right, bottom = padding
    cropped = image.crop((left, top, image.width - right, image.height - bottom))
    upscaled = cropped.resize(original_size, Image.Resampling.LANCZOS)
    sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return sharpened

# ========== Background Removal with Trim ==========
def remove_bg_cv(image_pil, white_thresh=240, trim_amount=1):
    image = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(255 - mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alpha_mask = np.zeros_like(gray)
    cv2.drawContours(alpha_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Trim edges using erosion
    if trim_amount > 0:
        kernel = np.ones((3, 3), np.uint8)
        alpha_mask = cv2.erode(alpha_mask, kernel, iterations=trim_amount)

    alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), sigmaX=0)
    b, g, r = cv2.split(image)
    rgba = cv2.merge((b, g, r, alpha_mask))
    return Image.fromarray(rgba)

# ========== BLIP Prompt ==========
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def suggest_prompt_from_image(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# ========== Load CosXL Pipeline ==========
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
edit_file = hf_hub_download(repo_id="stabilityai/cosxl", filename="cosxl_edit.safetensors")

pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file, num_in_channels=8, is_cosxl_edit=True, vae=vae, torch_dtype=torch.float16,
)
pipe_edit.scheduler = EDMEulerScheduler(
    sigma_min=0.002, sigma_max=120.0, sigma_data=1.0,
    prediction_type="v_prediction", sigma_schedule="exponential"
)
pipe_edit.to(device)
pipe_edit.enable_attention_slicing(1)
if hasattr(pipe_edit, "enable_xformers_memory_efficient_attention") and device == "cuda":
    pipe_edit.enable_xformers_memory_efficient_attention()

# ========== Inference ==========
@spaces.GPU
def run_edit(image, prompt, negative_prompt="", guidance_scale=7, steps=20, white_thresh=240, trim_amount=1, progress=gr.Progress(track_tqdm=True)):
    padded, original_size, padding = resize_and_pad(image)
    torch_gc()

    edited = pipe_edit(
        prompt=prompt,
        image=padded,
        height=488,
        width=488,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=min(steps, 20)
    ).images[0]
    torch_gc()

    aligned = crop_and_upscale(edited, original_size, padding)
    final = remove_bg_cv(aligned, white_thresh=white_thresh, trim_amount=trim_amount)

    filename = "output_edited.png"
    final.save(filename)
    return final, filename

# ========== Gradio UI ==========
css = '''
.gradio-container {
    max-width: 768px !important;
    margin: 0 auto;
}
'''

examples = [
    ["mountain.png", "make it a cloudy day"],
    ["painting.png", "make the earring fancier"]
]

with gr.Blocks(css=css) as demo:
    gr.Markdown('''# CosXL Edit (Smart Mask + Trim)
- Maintains alignment & size  
- Background removal with smart mask + edge trim  
- PNG download + BLIP prompt suggestion
''')

    with gr.Tab("CosXL Edit"):
        with gr.Group():
            image_edit = gr.Image(label="Upload Image", type="pil")
            with gr.Row():
                prompt_edit = gr.Textbox(show_label=False, scale=4, placeholder="Edit instructions")
                button_edit = gr.Button("Generate", min_width=120)
            output_edit = gr.Image(label="Result Image", interactive=False)
            download_button = gr.File(label="Download PNG")

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_edit = gr.Textbox(label="Negative Prompt")
                guidance_scale_edit = gr.Number(label="Guidance Scale", value=7, step=.5)
                steps_edit = gr.Slider(label="Steps (Max 20)", minimum=10, maximum=50, value=15, step=1)
                white_thresh_slider = gr.Slider(label="White Removal Threshold", minimum=220, maximum=255, value=240, step=1)
                trim_slider = gr.Slider(label="Trim Edge (px)", minimum=0, maximum=5, value=1, step=1)

        gr.Examples(
            examples=examples,
            fn=lambda img, p: run_edit(img, p)[:1],
            inputs=[image_edit, prompt_edit],
            outputs=[output_edit],
            cache_examples=False
        )

        def handle_image_upload(img):
            return gr.update(value=suggest_prompt_from_image(img))

        image_edit.change(fn=handle_image_upload, inputs=image_edit, outputs=prompt_edit)

        def trigger_edit(img, prompt, neg, scale, steps, white_thresh, trim_amount):
            final_img, file_path = run_edit(img, prompt, neg, scale, steps, white_thresh, trim_amount)
            return final_img, file_path

        button_edit.click(
            fn=trigger_edit,
            inputs=[
                image_edit, prompt_edit, negative_prompt_edit,
                guidance_scale_edit, steps_edit, white_thresh_slider, trim_slider
            ],
            outputs=[output_edit, download_button]
        )

        prompt_edit.submit(
            fn=trigger_edit,
            inputs=[
                image_edit, prompt_edit, negative_prompt_edit,
                guidance_scale_edit, steps_edit, white_thresh_slider, trim_slider
            ],
            outputs=[output_edit, download_button]
        )

if __name__ == "__main__":
    demo.launch(share=True)
