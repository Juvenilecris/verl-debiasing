# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Optional
from uuid import uuid4
import re
from PIL import Image
from verl.utils.reward_score import gsm8k

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WaterbirdInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of gsm8k.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the assistant.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "img_path":kwargs.pop('img_path', None)
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break

        self._instance_dict[instance_id]["response"] = content
        img_path=self._instance_dict[instance_id]['img_path']
        reward = await self.calculate_score(instance_id)
        
        # print(content)
        
        if reward == 1.0:
            box_pattern = re.compile(r"<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>")
            match = box_pattern.search(content)
            coords_str = match.groups()
            box_coords = [int(c) for c in coords_str] # -> [x1, y1, x2, y2]

            original_image = Image.open(img_path).convert("RGB")
            cropped_image = original_image.crop(box_coords)
            
            # 5. 將原始圖像和裁剪後的圖像打包到 additional_data 中
            multi_modal_data = {
                "image": [original_image, cropped_image]
            }
            additional_data = {
                "multimodal_data": multi_modal_data
            }

            response="""
* Step 2 Format: A JSON object with a single key ```json{"classification": "Result"}```

Step 2: Excellent. Now, using the original image provided first and the cropped image of the bird (based on your previous bounding box) provided second, classify the bird as either a "Waterbird" or a "Landbird".

Your answer must be in JSON format.

Original Image: <image>
Cropped Bird Image: <image>"""
            should_terminate_sequence=False
        else:
            response = "Your response don't match format"
            should_terminate_sequence = True
            additional_data={}

        return should_terminate_sequence, response, reward, additional_data

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        content=self._instance_dict[instance_id]["response"]
        box_pattern = re.compile(r"<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>")
        match = box_pattern.search(content)
        return 1.0 if match else 0.0


        # return gsm8k.compute_score(
        #     self._instance_dict[instance_id]["response"],
        #     self._instance_dict[instance_id]["ground_truth"],
        #     method="strict",
        #     format_score=0.0,
        #     score=1.0,
        # )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
