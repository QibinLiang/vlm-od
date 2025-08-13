# Copyright (c) 2025 Qibin Liang (RFAI tech) <physechan@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import yaml
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.utils.utils import torch_timer
from core.modules.object_detection import ObjectDetection

config = yaml.safe_load(open("configs/owlv2_single_step_object_detection.yaml", "r"))
od = ObjectDetection(config)
def detector(img, prompt, topk):
    return od(img, [prompt], topk)

@torch_timer(logger=None, extra_info="draw boxes on image")
def draw_boxes(image_pil, boxes, labels, scores, topk):
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()
    
    for box, label, score in list(zip(boxes, labels, scores))[:topk]:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)
    
    return image_pil

def detect_and_draw_owlv2(image_np: np.ndarray, prompt: str, topk:int) -> np.ndarray:
    image_pil = Image.fromarray(image_np)
    results = detector([image_pil], prompt, topk)
    boxes = results[0]["boxes"]
    labels = results[0]["labels"]
    scores = results[0]["scores"]
    image_with_boxes = draw_boxes(image_pil.copy(), boxes, labels, scores, topk)

    return np.array(image_with_boxes)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="A tongue")
            topk = gr.Slider(label="Top k", minimum=1, maximum=5, step=1, interactive=True)
            input_img = gr.Image(sources=["webcam"], type="numpy")
        with gr.Column():
            output_img = gr.Image(streaming=True)
        dep = input_img.stream(detect_and_draw_owlv2, [input_img, prompt, topk], [output_img],
                                time_limit=30, stream_every=0.3, concurrency_limit=30)

demo.launch()
# demo.launch(server_name="0.0.0.0", ssl_keyfile="cert.key", ssl_certfile="cert.pem", ssl_verify=False)

