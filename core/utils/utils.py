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
import time
import json

import torch

# Decorator to time a function using torch.cuda.Event. 
# requires a argument "logger" to be passed to the decorator
def torch_timer(logger=None, extra_info=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # if logger level is 25, then start the timer
            if logger and logger.isEnabledFor(25):
                if torch.cuda.is_available(): # gpu time
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    result = func(*args, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)
                    logger.verbose(f"Elapsed time - {extra_info} : {elapsed_time:.2f} ms")
                else: # cpu time
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    elapsed_time = (end_time - start_time) * 1000
                    logger.verbose(f"Elapsed time - {extra_info} : {elapsed_time:.2f} ms")
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

def export_object_detection_result(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def load_images_from_folder(folder_path, suffixes=['.png', '.jpg']):
    images = []
    for filename in os.listdir(folder_path):
        if any(filename.endswith(suffix) for suffix in suffixes):
            images.append(os.path.join(folder_path, filename))
        else:
            sub_folder_path = os.path.join(folder_path, filename)
            sub_folder_images = load_images_from_folder(sub_folder_path)
            images += sub_folder_images
    return images