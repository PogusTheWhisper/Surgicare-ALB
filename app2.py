import gradio as gr
import spaces
import torch
import os
import gc
import warnings
import numpy as np
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import (
    StableDiffusionXLInstructPix2PixPipeline,
    AutoencoderKL,
    EDMEulerScheduler,
)
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

# ========== Device Setup ==========
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
if device == "mps":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
elif device == "cuda":
    torch.backends.cudnn.benchmark = True

# ========== Memory Management ==========
def torch_gc():
    """Basic memory cleanup compatible with all PyTorch versions"""
    try:
        # Try to empty MPS cache if available
        if device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    except:
        pass
        
    # Always do GC collection
    gc.collect()
    
    # Try CUDA cleanup just in case
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
        
# ========== Image Utilities ==========
def resize_image(image, resolution):
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = resolution
        new_height = int((resolution / original_width) * original_height)
    else:
        new_height = resolution
        new_width = int((resolution / original_height) * original_width)
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

# ========== Prompt Suggestion (BLIP) ==========
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

torch_gc()

def suggest_prompt_from_image(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    torch_gc()
    return blip_processor.decode(out[0], skip_special_tokens=True)

# ========== Load CosXL Edit Pipeline ==========
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
edit_file = hf_hub_download(repo_id="stabilityai/cosxl", filename="cosxl_edit.safetensors")

torch_gc()

pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file, num_in_channels=8, is_cosxl_edit=True, vae=vae, torch_dtype=torch.float16,
)
pipe_edit.scheduler = EDMEulerScheduler(
    sigma_min=0.002, sigma_max=120.0, sigma_data=1.0,
    prediction_type="v_prediction", sigma_schedule="exponential"
)
pipe_edit.to(device)
if hasattr(pipe_edit, "enable_attention_slicing"):
    pipe_edit.enable_attention_slicing(1)
if hasattr(pipe_edit, "enable_xformers_memory_efficient_attention") and device == "cuda":
    pipe_edit.enable_xformers_memory_efficient_attention()

torch_gc()

# ========== Inference ==========
@spaces.GPU
def run_edit(image, prompt, negative_prompt="", guidance_scale=7, steps=20, progress=gr.Progress(track_tqdm=True)):
    steps = min(steps, 20)  # Clamp to avoid crash
    image = resize_image(image, 1024)
    print("Image resized to", image.size)
    width, height = image.size
    torch_gc()
    return pipe_edit(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]

# ========== Gradio UI ==========
css = '''
.gradio-container {
    max-width: 768px !important;
    margin: 0 auto;
}
'''

edit_examples = [
    ["mountain.png", "make it a cloudy day"],
    ["painting.png", "make the earring fancier"]
]

with gr.Blocks(css=css) as demo:
    gr.Markdown('''# CosXL Edit (Precision Optimized)  
    - Output keeps original size and position  
    - Background removed based on white threshold  
    - Prompt auto-suggested via BLIP  
    License: [Non-Commercial](https://huggingface.co/stabilityai/cosxl/blob/main/LICENSE)
    ''')

    with gr.Tab("CosXL Edit"):
        with gr.Group():
            image_edit = gr.Image(label="Upload Image", type="pil")
            with gr.Row():
                prompt_edit = gr.Textbox(show_label=False, scale=4, placeholder="Edit instructions, e.g.: Make the day cloudy")
                button_edit = gr.Button("Generate", min_width=120)
            output_edit = gr.Image(label="Result Image", interactive=False)

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_edit = gr.Textbox(label="Negative Prompt")
                guidance_scale_edit = gr.Number(label="Guidance Scale", value=7)
                steps_edit = gr.Slider(label="Steps (Max 20)", minimum=10, maximum=50, value=20)

        gr.Examples(
            examples=edit_examples,
            fn=run_edit,
            inputs=[image_edit, prompt_edit],
            outputs=[output_edit],
            cache_examples=True
        )

        def handle_image_upload(img):
            suggested = suggest_prompt_from_image(img)
            return gr.update(value=suggested)

        image_edit.change(fn=handle_image_upload, inputs=image_edit, outputs=prompt_edit)

        gr.on(
            triggers=[button_edit.click, prompt_edit.submit],
            fn=run_edit,
            inputs=[image_edit, prompt_edit, negative_prompt_edit, guidance_scale_edit, steps_edit],
            outputs=[output_edit]
        )

if __name__ == "__main__":
    demo.launch(share=True)
