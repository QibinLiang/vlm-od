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
import PIL.Image
import torchvision.transforms as transforms
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers.models.owlv2.modeling_owlv2 import Owlv2ObjectDetectionOutput

from core.models.base import *
from core.utils.logging import get_logger
from core.utils.utils import torch_timer
from core.models.owl.modeling_owlv2 import Owlv2ForObjectDetection as Owlv2ForObjectDetectionFA2

logger = get_logger(__name__)

class Owlv2ProcessorFast(Owlv2Processor):
    # for accelerating inference in preprocessing
    def owlv2_img_preprocess(self, images, use_cuda=True):
        # if images is a list of PIL images, convert them to tensors
        if isinstance(images[0], PIL.Image.Image):
            images = [transforms.ToTensor()(image) for image in images]
        if use_cuda:
            # get current device
            device = torch.cuda.current_device()
            images = [image.to(device) for image in images]
        scale: float = 0.00392156862745098
        # scale
        # images = [image * scale for image in images]
        # pad all images to the same size
        images = [torch.nn.functional.pad(
            image,
            (0, max(image.shape) - image.shape[2], 0, max(image.shape) - image.shape[1]),
            value=0.5) for image in images]
        # resize images 
        images = [transforms.Resize((960, 960))(image) for image in images]
        # normalize images
        images = [transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073], 
                                    std=[0.26862954,0.26130258,0.27577711])(image) for image in images]
        return images
    
    def __call__(self, 
                 text,
                 images,
                 return_tensors="pt"):
        owl_processor_output = super().__call__(
            text=text,
            return_tensors=return_tensors
        )
        images_output = self.owlv2_img_preprocess(images)
        owl_processor_output["pixel_values"] = torch.stack(images_output)
        return owl_processor_output

class OWLV2ONNX:
    def __init__(self, onnx_path, onnx_optimize):
        self.onnx_path = onnx_path
        session_options = ort.SessionOptions()
        if onnx_optimize:
            logger.verbose("using onnx optimizations....")
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, 
                                            providers=['CUDAExecutionProvider'],
                                            sess_options=session_options)
        logger.verbose(f"Loaded OWLV2 ONNX model from {onnx_path}")
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

    def eval(self):
        return 

    def to(self, device):
        logger.verbose(f"OWLV2 ONNX runs on GPU by default")
    
    def __call__(self, pixel_values, input_ids, attention_mask):
        inputs = {
            "pixel_values": pixel_values.cpu().numpy(),
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy()
        }
        outputs = self.session.run(self.output_names, inputs)
        # Convert outputs to tensors
        outputs = {name: torch.tensor(output) for name, output in zip(self.output_names, outputs)}
        outputs = Owlv2ObjectDetectionOutput(**outputs)
        return outputs

class OWLV2(Model, CVFunctional):
    def __init__(self, config):
        super().__init__()
        self.config = config
        logger.verbose("OWLV2 init")
        self.processor = Owlv2ProcessorFast.from_pretrained(config["model_name"])
        logger.verbose("Using faster owlv2 image processor")
        if config.get("use_onnx"):
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("onnxruntime is not installed")
            logger.verbose("Loading OWLV2 ONNX model")
            if config.get("use_optimized_model"):
                logger.verbose("Loading the optimized model")
                self.model = OWLV2ONNX(config["onnx_optimized_path"], config["onnx_optimize"])
            else:
                self.model = OWLV2ONNX(config["onnx_path"], config["onnx_optimize"])
        else:
            logger.verbose("Loading OWLV2 model from transformers")
            if config["use_fa2"]:
                OWLOD = Owlv2ForObjectDetectionFA2
                logger.verbose("using flash attention 2")
            else:
                OWLOD = Owlv2ForObjectDetection
                logger.verbose("using naive attention")
            self.model = OWLOD.from_pretrained(config["model_name"], 
                                               device_map=config["model_device"],
                                               torch_dtype=torch.float16 if config["dtype"]=='fp16' else torch.float32)
                                               #attn_implementation=config["attn_implementation"])
        self.model.eval()
        self.device = config["model_device"]
        self.set_device(self.device)

    @torch_timer(logger, extra_info="owlv2 model infernece")
    def inference(self, images, prompts, threshold, topk=-1):
        with torch.no_grad():
            if len(prompts) < len(images):
                prompts = [prompts[0]] * len(images)
            inputs = self.processor(text=prompts, 
                                    images=images, 
                                    return_tensors="pt").to(self.device)
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
        
    def __call__(self, images_paths, prompts, owlv2_threshold, topk):
        images = [self.load_image(image_path) for image_path in images_paths]
        return self.inference(images, prompts, owlv2_threshold, topk)[0]
