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

import cv2
import numpy as np

def draw_img_with_boxes(image, boxes, output_file, labels=None, scores=None, ):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Draw boxes on the image
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box]
        cv2.rectangle(opencv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.putText(opencv_image, f"{label}: {round(score, 3)}",
                    (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Save the image with bounding boxes
    cv2.imwrite(output_file, opencv_image)