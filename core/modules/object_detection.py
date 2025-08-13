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

from core.models import *
from core.utils.logging import get_logger

logger = get_logger(__name__)

OBJECT_DETECTION_MODELS= {
    "owlv2": OWLV2,
    "grounding-dino": GroundingDino
}

class ObjectDetection:
    def __init__(self, config):
        logger.verbose("ObjectDetection init")
        logger.verbose(config)
        model_type = config["object_detection_model"]
        model_config = config["model_config"]
        if model_type not in OBJECT_DETECTION_MODELS:
            raise ValueError(f"Model {model_type} not supported.")
        self.model = OBJECT_DETECTION_MODELS[model_type](model_config)
        self.strategy = model_config["strategy"]
        if self.strategy not in ["single", "dual"]:
            raise ValueError(f"Strategy {self.strategy} not supported.")
        if self.strategy == "dual":
            self.step1_threshold = model_config["step1_threshold"]
            self.step1_topk = model_config["step1_topk"]
            self.step1_prompts = model_config["step1_prompts"]
            self.step1_boxes_expansion = model_config["step1_boxes_expansion"]

            self.step2_threshold = model_config["step2_threshold"]
            self.step2_topk = model_config["step2_topk"]
            self.step2_prompts = model_config["step2_prompts"]
            self.topk = model_config["topk"]
        else:
            self.threshold = model_config["threshold"]
            self.topk = model_config["topk"]
            self.prompts = model_config["prompts"]

    @torch_timer(logger, extra_info="object detection with dual step")  
    def _single_step(self, image_paths, prompt=None, topk=None):
        if prompt is None:
            prompt = self.prompts
        if topk is None:
            topk = self.topk
        # if the given image_paths is a list of strings, load the images
        if isinstance(image_paths[0], str):
            image_paths = [self.model.load_image(image_path) for image_path in image_paths]
        else:
            image_paths = image_paths
        # model.inference returns [bboxes, images_size], we only care about bboxes
        result = self.model.inference(image_paths, prompt, self.threshold, topk)[0]
        # to list
        for i in range(len(result)):
            result[i]["boxes"] = result[i]["boxes"].cpu().tolist()
            result[i]["scores"] = result[i]["scores"].cpu().tolist()
            if isinstance(result[i]["labels"], str):
                result[i]["labels"] = [result[i]["labels"]]
            else:
                result[i]["labels"] = result[i]["labels"].cpu().tolist()
        return result

    def __call__(self, image_paths, prompt, topk):
        if self.strategy == "single":
            return self._single_step(image_paths, prompt, topk)
        else:
            raise ValueError(f"Strategy {self.strategy} not supported.")
