from mmgp import offload
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel
import numpy as np
import json
import os

from src.flux.condition import Condition
from src.flux.generate import seed_everything, generate
from src.nsfw import zero_if_nsfw


pipe = None
use_int8 = False
quantize_transformer = False
CONFIG_FILE = "advanced_config.json"


def load_config():
    config = {
        "num_steps": 8,
        "use_int8": False,
        "quantize_transformer": False,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            config.update(data)
        except Exception:
            pass
    return config


def save_config(num_steps, use_int8_val, quant_val):
    with open(CONFIG_FILE, "w") as f:
        json.dump(
            {
                "num_steps": int(num_steps),
                "use_int8": bool(use_int8_val),
                "quantize_transformer": bool(quant_val),
            },
            f,
        )


config = load_config()
use_int8 = config["use_int8"]
quantize_transformer = config["quantize_transformer"]
num_steps_default = config["num_steps"]


def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def init_pipeline(token=None):
    global pipe
    if False and (use_int8 or get_gpu_memory() < 33):
        transformer_model = FluxTransformer2DModel.from_pretrained(
            "sayakpaul/flux.1-schell-int8wo-improved",
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
            token=token,
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer_model,
            torch_dtype=torch.bfloat16,
            token=token,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, token=token
        )
    pipe = pipe.to("cpu")
    pipe.load_lora_weights(
        "Yuanshi/OminiControl",
        weight_name="omini/subject_512.safetensors",
        adapter_name="subject",
        token=token,
    )
    offload.profile(
        pipe,
        profile_no=int(args.profile),
        verboseLevel=int(args.verbose),
        quantizeTransformer=quantize_transformer,
    )

def process_image_and_text(image, text, num_steps, use_int8_val, quant_val):
    global use_int8, quantize_transformer
    use_int8 = bool(use_int8_val)
    quantize_transformer = bool(quant_val)
    save_config(num_steps, use_int8, quantize_transformer)
    conditions = None
    if image is not None:
        # center crop image
        w, h, min_size = image.size[0], image.size[1], min(image.size)
        image = image.crop(
            (
                (w - min_size) // 2,
                (h - min_size) // 2,
                (w + min_size) // 2,
                (h + min_size) // 2,
            )
        )
        image = image.resize((512, 512))

        condition = Condition("subject", image, position_delta=(0, 32))
        conditions = [condition]

    if pipe is None:
        init_pipeline(token=args.token)

    result_img = generate(
        pipe,
        prompt=text.strip(),
        conditions=conditions,
        num_inference_steps=int(num_steps),
        height=512,
        width=512,
    ).images[0]
    result_img = zero_if_nsfw(result_img)

    return result_img


def get_samples():
    sample_list = [
        {
            "image": "assets/oranges.jpg",
            "text": "A very close up view of this item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. With text on the screen that reads 'Omini Control!'",
        },
        {
            "image": "assets/penguin.jpg",
            "text": "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat, holding a sign that reads 'Omini Control!'",
        },
        {
            "image": "assets/rc_car.jpg",
            "text": "A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground.",
        },
        {
            "image": "assets/clock.jpg",
            "text": "In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
        },
        {
            "image": "assets/tshirt.jpg",
            "text": "On the beach, a lady sits under a beach umbrella with 'Omini' written on it. She's wearing this shirt and has a big smile on her face, with her surfboard hehind her.",
        },
    ]
    return [[Image.open(sample["image"]), sample["text"]] for sample in sample_list]


with gr.Blocks() as demo:
    gr.Markdown("## Hackathon Image Generator")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil")
            input_text = gr.Textbox(lines=2)
            advanced_toggle = gr.Checkbox(label="Advanced", value=False)
            with gr.Column(visible=False) as advanced_opts:
                steps_slider = gr.Slider(
                    1,
                    50,
                    value=num_steps_default,
                    step=1,
                    label="Number of Sampling Steps",
                )
                int8_checkbox = gr.Checkbox(label="Use INT8", value=use_int8)
                quant_checkbox = gr.Checkbox(
                    label="Quantize Transformer", value=quantize_transformer
                )
        with gr.Column():
            output_image = gr.Image(type="pil")

    run_button = gr.Button("Generate")
    advanced_toggle.change(
        lambda v: gr.update(visible=v), advanced_toggle, advanced_opts
    )
    run_button.click(
        process_image_and_text,
        inputs=[
            input_image,
            input_text,
            steps_slider,
            int8_checkbox,
            quant_checkbox,
        ],
        outputs=output_image,
    )
    gr.Examples(get_samples(), inputs=[input_image, input_text])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")
    parser.add_argument('--token', type=str, default=None, help="HuggingFace access token")
    parser.add_argument('--server-name', type=str, default="0.0.0.0", dest="server_name", help="Server name for gradio")
    parser.add_argument('--server-port', type=int, default=7860, dest="server_port", help="Server port for gradio")

    args = parser.parse_args()

    init_pipeline(token=args.token)
    demo.launch(
        debug=True,
        server_name=args.server_name,
        server_port=args.server_port,
    )
