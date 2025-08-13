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

import yaml
from core.modules.object_detection import ObjectDetection
from core.utils.draw import draw_img_with_boxes
from core.utils.utils import export_object_detection_result, load_images_from_folder


class ObjectDetectionPipeline:
    def __init__(self, config_file):
        self.config = yaml.safe_load(open(config_file, "r"))
        self.module = ObjectDetection(self.config)
        self.input_folder = self.config["input_folder"]
        self.output_folder = self.config["output_folder"]
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def run(self, files):
        return self.module(files), files
        
    def run_all_by_batch(self):
        files = self.load_images_from_folder(self.config["input_folder"])
        batch_size = self.config["batch_size"]
        if batch_size > 0:
            for i in range(0, len(files), batch_size):
                files_batch = files[i:i + batch_size]
                yield self.module(files_batch), files_batch
    
    def export_result_to_json(self, result, output_file):
        export_object_detection_result(result, output_file)

    def load_images_from_folder(self, folder_path, suffixes=['.png', '.jpg']):
        return load_images_from_folder(folder_path, suffixes)
    
    def draw_img(self, image, boxes, output_file, labels=None, scores=None, output_folder=None):
        if output_folder is None:
            output_file = os.path.join(self.output_folder, output_file)
        draw_img_with_boxes(image, boxes, output_file, labels, scores)