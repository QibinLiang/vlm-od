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

from PIL import Image

class Model:
    def __init__(self):
        self.device = "cpu"
        
    def cuda(self):
        self.device = "cuda"
        self.model.cuda()

    def cpu(self):
        self.device = "cpu"
        self.model.cpu()

    def set_device(self, device):
        self.to(device)

    def to(self, device):
        self.device = device
        self.model.to(device)    

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Model call not implemented")
    
class CVFunctional:

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image
