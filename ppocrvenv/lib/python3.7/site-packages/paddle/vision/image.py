#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from PIL import Image
from paddle.utils import try_import

__all__ = []

_image_backend = 'pil'


def set_image_backend(backend):
    """
    Specifies the backend used to load images in class ``paddle.vision.datasets.ImageFolder`` 
    and ``paddle.vision.datasets.DatasetFolder`` . Now support backends are pillow and opencv. 
    If backend not set, will use 'pil' as default. 

    Args:
        backend (str): Name of the image load backend, should be one of {'pil', 'cv2'}.

    Examples:
    
        .. code-block:: python

            import os
            import shutil
            import tempfile
            import numpy as np
            from PIL import Image

            from paddle.vision import DatasetFolder
            from paddle.vision import set_image_backend

            set_image_backend('pil')

            def make_fake_dir():
                data_dir = tempfile.mkdtemp()

                for i in range(2):
                    sub_dir = os.path.join(data_dir, 'class_' + str(i))
                    if not os.path.exists(sub_dir):
                        os.makedirs(sub_dir)
                    for j in range(2):
                        fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype('uint8'))
                        fake_img.save(os.path.join(sub_dir, str(j) + '.png'))
                return data_dir

            temp_dir = make_fake_dir()

            pil_data_folder = DatasetFolder(temp_dir)

            for items in pil_data_folder:
                break

            # should get PIL.Image.Image
            print(type(items[0]))

            # use opencv as backend
            # set_image_backend('cv2')

            # cv2_data_folder = DatasetFolder(temp_dir)

            # for items in cv2_data_folder:
            #     break

            # should get numpy.ndarray
            # print(type(items[0]))

            shutil.rmtree(temp_dir)
    """
    global _image_backend
    if backend not in ['pil', 'cv2', 'tensor']:
        raise ValueError(
            "Expected backend are one of ['pil', 'cv2', 'tensor'], but got {}"
            .format(backend))
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images

    Returns:
        str: backend of image load.

    Examples:
    
        .. code-block:: python

            from paddle.vision import get_image_backend

            backend = get_image_backend()
            print(backend)

    """
    return _image_backend


def image_load(path, backend=None):
    """Load an image.

    Args:
        path (str): Path of the image.
        backend (str, optional): The image decoding backend type. Options are
            `cv2`, `pil`, `None`. If backend is None, the global _imread_backend 
            specified by ``paddle.vision.set_image_backend`` will be used. Default: None.

    Returns:
        PIL.Image or np.array: Loaded image.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision import image_load, set_image_backend

            fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype('uint8'))

            path = 'temp.png'
            fake_img.save(path)

            set_image_backend('pil')
            
            pil_img = image_load(path).convert('RGB')

            # should be PIL.Image.Image
            print(type(pil_img))

            # use opencv as backend
            # set_image_backend('cv2')

            # np_img = image_load(path)
            # # should get numpy.ndarray
            # print(type(np_img))
    
    """

    if backend is None:
        backend = _image_backend
    if backend not in ['pil', 'cv2', 'tensor']:
        raise ValueError(
            "Expected backend are one of ['pil', 'cv2', 'tensor'], but got {}"
            .format(backend))

    if backend == 'pil':
        return Image.open(path)
    elif backend == 'cv2':
        cv2 = try_import('cv2')
        return cv2.imread(path)
