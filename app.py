import gradio as gr
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler, StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
import numpy as np
import math
from PIL import Image

# ========== Download Weights ==========
edit_file = hf_hub_download(repo_id="stabilityai/cosxl", filename="cosxl_edit.safetensors")

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

# ========== Load Pipelines ==========
pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file, num_in_channels=8, is_cosxl_edit=True, vae=vae, torch_dtype=torch.float16,
)
pipe_edit.scheduler = EDMEulerScheduler(
    sigma_min=0.002, sigma_max=120.0, sigma_data=1.0,
    prediction_type="v_prediction", sigma_schedule="exponential"
)
pipe_edit.to("mps")

# ========== Inference Functions ==========

@spaces.GPU
def run_edit(image, prompt, negative_prompt="", guidance_scale=7, steps=20, progress=gr.Progress(track_tqdm=True)):
    steps = min(steps, 20)  # Clamp to avoid crash
    image = resize_image(image, 1024)
    print("Image resized to", image.size)
    width, height = image.size
    return pipe_edit(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]

# ========== UI ==========
css = '''
.gradio-container{
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

        gr.Examples(examples=edit_examples, fn=run_edit, inputs=[image_edit, prompt_edit], outputs=[output_edit], cache_examples=True)

    gr.on(
        triggers=[button_edit.click, prompt_edit.submit],
        fn=run_edit,
        inputs=[image_edit, prompt_edit, negative_prompt_edit, guidance_scale_edit, steps_edit],
        outputs=[output_edit]
    )

if __name__ == "__main__":
    demo.launch(share=True)
