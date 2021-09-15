# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from paddle.vision.transforms import ColorJitter as pp_ColorJitter

__all__  = ['ColorJitter']

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,**kwargs):
        self.aug = pp_ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data):
        image = data['image']
        image = self.aug(image)
        data['image'] = image
        return data
