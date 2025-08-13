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

import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

from core.models.base import *
from core.utils.logging import get_logger
from core.utils.utils import torch_timer

logger = get_logger(__name__)


class GroundingDino(Model, CVFunctional):
    def __init__(self, config):
        super().__init__()
        self.config = config
        logger.verbose("GroundingDino init")
        self.processor = GroundingDinoProcessor.from_pretrained(config["model_name"], use_fast=True)
        logger.verbose("Using faster GroundingDino image processor")
        if config.get("use_onnx"):
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("onnxruntime is not installed")
            logger.error("Grounding dino does not support ONNX now. The torch implementation will be used instead.")
        else:
            logger.verbose("Loading GroundingDino model from transformers")
            if config["use_fa2"]:
                DINO = GroundingDinoForObjectDetection
                logger.verbose("using flash attention 2")
            else:
                DINO = GroundingDinoForObjectDetection
                logger.verbose("using naive attention")
            self.model = DINO.from_pretrained(config["model_name"], 
                                               device_map=config["model_device"],
                                               torch_dtype=torch.float16 if config["dtype"]=='fp16' else torch.float32)
                                               #attn_implementation=config["attn_implementation"])
        self.model.eval()
        self.device = config["model_device"]
        self.set_device(self.device)
        self.dtype = torch.float16 if config["dtype"]=='fp16' else torch.float32

    @torch_timer(logger, extra_info="grounding dino model infernece")
    def inference(self, images, prompts, threshold, topk=-1):
        with torch.no_grad():
            if len(prompts) < len(images):
                prompts = [prompts[0]] * len(images)
            inputs = self.processor(text=prompts, 
                                    images=images, 
                                    return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            outputs = self.model(**inputs)
            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1] for image in images])
            # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
            results = self.processor.post_process_grounded_object_detection(outputs=outputs, 
                                                                            target_sizes=target_sizes, 
                                                                            threshold=threshold)
            if topk > 0:
                for i in range(len(results)):
                    indices = results[i]["scores"].argsort(descending=True)[:topk]
                    results[i]["boxes"] = results[i]["boxes"][indices]
                    results[i]["scores"] = results[i]["scores"][indices]
                    results[i]["labels"] = results[i]["labels"][indices]
            return results, target_sizes
        
    def __call__(self, images_paths, prompts, dino_threshold, topk):
        images = [self.load_image(image_path) for image_path in images_paths]
        return self.inference(images, prompts, dino_threshold, topk)[0]
