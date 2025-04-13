import gradio as gr
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler, StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import math
import os
import gc
from PIL import Image

# ========== Device Setup ==========
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
if device == "mps":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
elif device == "cuda":
    torch.backends.cudnn.benchmark = True

# ========== Download Weights ==========
edit_file = hf_hub_download(repo_id="stabilityai/cosxl", filename="cosxl_edit.safetensors")

# ========== Memory Management ==========
def torch_gc():
    try:
        if device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    except:
        pass
    gc.collect()
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

# ========== Image Resize ==========
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

# ========== Load VAE ==========
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# ========== Load Edit Pipeline ==========
pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file,
    num_in_channels=8,
    is_cosxl_edit=True,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe_edit.scheduler = EDMEulerScheduler(
    sigma_min=0.002, sigma_max=120.0, sigma_data=1.0,
    prediction_type="v_prediction", sigma_schedule="exponential"
)
pipe_edit.to(device)
torch_gc()

# ========== Load BLIP Captioning ==========
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
torch_gc()

def generate_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    torch_gc()
    return caption

# ========== Inference Function ==========
@spaces.GPU
def run_edit(image, prompt, negative_prompt="", guidance_scale=7, steps=20, progress=gr.Progress(track_tqdm=True)):
    steps = min(steps, 20)
    image = resize_image(image, 1024)
    print("Image resized to", image.size)
    width, height = image.size
    torch_gc()
    result = pipe_edit(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    )
    return result.images[0]

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
    gr.Markdown('''# CosXL demo  
    Unofficial demo for CosXL, a SDXL model tuned to produce full color range images. CosXL Edit allows you to perform edits on images.  
    Both have a [non-commercial community license](https://huggingface.co/stabilityai/cosxl/blob/main/LICENSE)
    ''')

    with gr.Tab("CosXL Edit"):
        with gr.Group():
            image_edit = gr.Image(label="Image you would like to edit", type="pil")
            with gr.Row():
                prompt_edit = gr.Textbox(show_label=False, scale=4, placeholder="Edit instructions, e.g.: Make the day cloudy")
                button_edit = gr.Button("Generate", min_width=120)
            output_edit = gr.Image(label="Your result image", interactive=False)
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_edit = gr.Textbox(label="Negative Prompt")
                guidance_scale_edit = gr.Number(label="Guidance Scale", value=7)
                steps_edit = gr.Slider(label="Steps (Max 20 for CosXL Edit)", minimum=10, maximum=50, value=20)

        # Auto-fill prompt from BLIP when image is uploaded
        image_edit.change(fn=generate_caption, inputs=[image_edit], outputs=[prompt_edit])

        gr.Examples(examples=edit_examples, fn=run_edit, inputs=[image_edit, prompt_edit], outputs=[output_edit], cache_examples=True)

    gr.on(
        triggers=[button_edit.click, prompt_edit.submit],
        fn=run_edit,
        inputs=[image_edit, prompt_edit, negative_prompt_edit, guidance_scale_edit, steps_edit],
        outputs=[output_edit]
    )

if __name__ == "__main__":
    demo.launch(share=True)
