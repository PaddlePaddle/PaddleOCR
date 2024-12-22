# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

"""
This code refers to PaddleClas\ppcls\arch\backbone\legendary_models\swin_transformer.py,
                         PaddleClas\ppcls\arch\backbone\base\theseus_layer.py
"""

import os
import time
from tqdm import tqdm
import hashlib
import requests
import shutil
import tarfile
import zipfile
import numpy as np
from typing import Tuple, List, Dict, Union, Callable, Any

import paddle
import paddle.nn as nn
from ppocr.utils.logging import get_logger
from paddle.nn.initializer import TruncatedNormal, Normal, Constant
import paddle.nn.functional as F


logger = get_logger()


MODEL_URLS = {
    "SwinTransformer_tiny_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_small_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_small_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_base_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_base_patch4_window12_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window12_384_pretrained.pdparams",
    "SwinTransformer_large_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_large_patch4_window12_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window12_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


WEIGHTS_HOME = os.path.expanduser("~/.paddleclas/weights")


Linear = nn.Linear


trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def to_2tuple(x):
    return tuple([x] * 2)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor.floor_()  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.res_dict = {}
        self.res_name = self.full_name()
        self.pruner = None
        self.quanter = None

        self.init_net(*args, **kwargs)

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"logits": output}
        # 'list' is needed to avoid error raised by popping self.res_dict
        for res_key in list(self.res_dict):
            # clear the res_dict because the forward process may change according to input
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def init_net(
        self,
        stages_pattern=None,
        return_patterns=None,
        return_stages=None,
        freeze_befor=None,
        stop_after=None,
        *args,
        **kwargs,
    ):
        # init the output of net
        if return_patterns or return_stages:
            if return_patterns and return_stages:
                msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
                logger.warning(msg)
                return_stages = None

            if return_stages is True:
                return_patterns = stages_pattern

            # return_stages is int or bool
            if type(return_stages) is int:
                return_stages = [return_stages]
            if isinstance(return_stages, list):
                if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                    msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                    logger.warning(msg)
                    return_stages = [
                        val
                        for val in return_stages
                        if val >= 0 and val < len(stages_pattern)
                    ]
                return_patterns = [stages_pattern[i] for i in return_stages]

            if return_patterns:
                # call update_res function after the __init__ of the object has completed execution, that is, the contructing of layer or model has been completed.
                def update_res_hook(layer, input):
                    self.update_res(return_patterns)

                self.register_forward_pre_hook(update_res_hook)

        # freeze subnet
        if freeze_befor is not None:
            self.freeze_befor(freeze_befor)

        # set subnet to Identity
        if stop_after is not None:
            self.stop_after(stop_after)

    def init_res(self, stages_pattern, return_patterns=None, return_stages=None):
        msg = '"init_res" will be deprecated, please use "init_net" instead.'
        logger.warning(DeprecationWarning(msg))

        if return_patterns and return_stages:
            msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
            logger.warning(msg)
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern
        # return_stages is int or bool
        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                logger.warning(msg)
                return_stages = [
                    val
                    for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

        if return_patterns:
            self.update_res(return_patterns)

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "The function 'replace_sub()' is deprecated, please use 'upgrade_sublayer()' instead."
        logger.error(DeprecationWarning(msg))
        raise DeprecationWarning(msg)

    def upgrade_sublayer(
        self,
        layer_name_pattern: Union[str, List[str]],
        handle_func: Callable[[nn.Layer, str], nn.Layer],
    ) -> Dict[str, nn.Layer]:
        """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.

        Args:
            layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
            handle_func (Callable[[nn.Layer, str], nn.Layer]): The function to modify target layer specified by 'layer_name_pattern'. The formal params are the layer(nn.Layer) and pattern(str) that is (a member of) layer_name_pattern (when layer_name_pattern is List type). And the return is the layer processed.

        Returns:
            Dict[str, nn.Layer]: The key is the pattern and corresponding value is the result returned by 'handle_func()'.

        Examples:

            from paddle import nn
            import paddleclas

            def rep_func(layer: nn.Layer, pattern: str):
                new_layer = nn.Conv2D(
                    in_channels=layer._in_channels,
                    out_channels=layer._out_channels,
                    kernel_size=5,
                    padding=2
                )
                return new_layer

            net = paddleclas.MobileNetV1()
            res = net.upgrade_sublayer(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
            print(res)
            # {'blocks[11].depthwise_conv.conv': the corresponding new_layer, 'blocks[12].depthwise_conv.conv': the corresponding new_layer}
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        hit_layer_pattern_list = []
        for pattern in layer_name_pattern:
            # parse pattern to find target layer and its parent
            layer_list = parse_pattern_str(pattern=pattern, parent_layer=self)
            if not layer_list:
                continue

            sub_layer_parent = layer_list[-2]["layer"] if len(layer_list) > 1 else self
            sub_layer = layer_list[-1]["layer"]
            sub_layer_name = layer_list[-1]["name"]
            sub_layer_index_list = layer_list[-1]["index_list"]

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index_list:
                if len(sub_layer_index_list) > 1:
                    sub_layer_parent = getattr(sub_layer_parent, sub_layer_name)[
                        sub_layer_index_list[0]
                    ]
                    for sub_layer_index in sub_layer_index_list[1:-1]:
                        sub_layer_parent = sub_layer_parent[sub_layer_index]
                    sub_layer_parent[sub_layer_index_list[-1]] = new_sub_layer
                else:
                    getattr(sub_layer_parent, sub_layer_name)[
                        sub_layer_index_list[0]
                    ] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            hit_layer_pattern_list.append(pattern)
        return hit_layer_pattern_list

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        layer_list = parse_pattern_str(stop_layer_name, self)
        if not layer_list:
            return False

        parent_layer = self
        for layer_dict in layer_list:
            name, index_list = layer_dict["name"], layer_dict["index_list"]
            if not set_identity(parent_layer, name, index_list):
                msg = f"Failed to set the layers that after stop_layer_name('{stop_layer_name}') to IdentityLayer. The error layer's name is '{name}'."
                logger.warning(msg)
                return False
            parent_layer = layer_dict["layer"]

        return True

    def freeze_befor(self, layer_name: str) -> bool:
        """freeze the layer named layer_name and its previous layer.

        Args:
            layer_name (str): The name of layer that would be freezed.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        def stop_grad(layer, pattern):
            class StopGradLayer(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    x = self.layer(x)
                    x.stop_gradient = True
                    return x

            new_layer = StopGradLayer()
            return new_layer

        res = self.upgrade_sublayer(layer_name, stop_grad)
        if len(res) == 0:
            msg = "Failed to stop the gradient befor the layer named '{layer_name}'"
            logger.warning(msg)
            return False
        return True

    def update_res(self, return_patterns: Union[str, List[str]]) -> Dict[str, nn.Layer]:
        """update the result(s) to be returned.

        Args:
            return_patterns (Union[str, List[str]]): The name of layer to return output.

        Returns:
            Dict[str, nn.Layer]: The pattern(str) and corresponding layer(nn.Layer) that have been set successfully.
        """

        # clear res_dict that could have been set
        self.res_dict = {}

        class Handler(object):
            def __init__(self, res_dict):
                # res_dict is a reference
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                if hasattr(layer, "hook_remove_helper"):
                    layer.hook_remove_helper.remove()
                layer.hook_remove_helper = layer.register_forward_post_hook(
                    save_sub_res_hook
                )
                return layer

        handle_func = Handler(self.res_dict)

        hit_layer_pattern_list = self.upgrade_sublayer(
            return_patterns, handle_func=handle_func
        )

        if hasattr(self, "hook_remove_helper"):
            self.hook_remove_helper.remove()
        self.hook_remove_helper = self.register_forward_post_hook(
            self._return_dict_hook
        )

        return hit_layer_pattern_list


def save_sub_res_hook(layer, input, output):
    layer.res_dict[layer.res_name] = output


def set_identity(
    parent_layer: nn.Layer, layer_name: str, layer_index_list: str = None
) -> bool:
    """set the layer specified by layer_name and layer_index_list to Indentity.

    Args:
        parent_layer (nn.Layer): The parent layer of target layer specified by layer_name and layer_index_list.
        layer_name (str): The name of target layer to be set to Indentity.
        layer_index_list (str, optional): The index of target layer to be set to Indentity in parent_layer. Defaults to None.

    Returns:
        bool: True if successfully, False otherwise.
    """

    stop_after = False
    for sub_layer_name in parent_layer._sub_layers:
        if stop_after:
            parent_layer._sub_layers[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index_list and stop_after:
        layer_container = parent_layer._sub_layers[layer_name]
        for num, layer_index in enumerate(layer_index_list):
            stop_after = False
            for i in range(num):
                layer_container = layer_container[layer_index_list[i]]
            for sub_layer_index in layer_container._sub_layers:
                if stop_after:
                    parent_layer._sub_layers[layer_name][sub_layer_index] = Identity()
                    continue
                if layer_index == sub_layer_index:
                    stop_after = True

    return stop_after


def parse_pattern_str(
    pattern: str, parent_layer: nn.Layer
) -> Union[None, List[Dict[str, Union[nn.Layer, str, None]]]]:
    """parse the string type pattern.

    Args:
        pattern (str): The pattern to discribe layer.
        parent_layer (nn.Layer): The root layer relative to the pattern.

    Returns:
        Union[None, List[Dict[str, Union[nn.Layer, str, None]]]]: None if failed. If successfully, the members are layers parsed in order:
                                                                [
                                                                    {"layer": first layer, "name": first layer's name parsed, "index": first layer's index parsed if exist},
                                                                    {"layer": second layer, "name": second layer's name parsed, "index": second layer's index parsed if exist},
                                                                    ...
                                                                ]
    """

    pattern_list = pattern.split(".")
    if not pattern_list:
        msg = f"The pattern('{pattern}') is illegal. Please check and retry."
        logger.warning(msg)
        return None

    layer_list = []
    while len(pattern_list) > 0:
        if "[" in pattern_list[0]:
            target_layer_name = pattern_list[0].split("[")[0]
            target_layer_index_list = list(
                index.split("]")[0] for index in pattern_list[0].split("[")[1:]
            )
        else:
            target_layer_name = pattern_list[0]
            target_layer_index_list = None

        target_layer = getattr(parent_layer, target_layer_name, None)

        if target_layer is None:
            msg = f"Not found layer named('{target_layer_name}') specifed in pattern('{pattern}')."
            logger.warning(msg)
            return None

        if target_layer_index_list:
            for target_layer_index in target_layer_index_list:
                if int(target_layer_index) < 0 or int(target_layer_index) >= len(
                    target_layer
                ):
                    msg = f"Not found layer by index('{target_layer_index}') specifed in pattern('{pattern}'). The index should < {len(target_layer)} and > 0."
                    logger.warning(msg)
                    return None
                target_layer = target_layer[target_layer_index]

        layer_list.append(
            {
                "layer": target_layer,
                "name": target_layer_name,
                "index_list": target_layer_index_list,
            }
        )

        pattern_list = pattern_list[1:]
        parent_layer = target_layer

    return layer_list


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def check_support_fused_op(use_fused_linear):
    if use_fused_linear:
        if paddle.device.cuda.get_device_capability()[0] >= 8:
            return True
        else:
            logger.warning(
                "The current device don't support Fused OP! Using the general Linear instead."
            )
    return False


def _set_ssld_pretrained(
    pretrained_path, use_ssld=False, use_ssld_stage1_pretrained=False
):
    if use_ssld and "ssld" not in pretrained_path:
        pretrained_path = pretrained_path.replace("_pretrained", "_ssld_pretrained")
    if use_ssld_stage1_pretrained and "ssld" in pretrained_path:
        pretrained_path = pretrained_path.replace(
            "ssld_pretrained", "ssld_stage1_pretrained"
        )
    return pretrained_path


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith("http://") or path.startswith("https://")


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = os.path.split(url)[-1]
    fpath = fname
    return os.path.join(root_dir, fpath)


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info(
            "File {} md5 check failed, {}(calc) != "
            "{}(base)".format(fullname, calc_md5sum, md5sum)
        )
        return False
    return True


DOWNLOAD_RETRY_LIMIT = 3


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not os.path.exists(path):
        os.makedirs(path)

    fname = os.path.split(url)[-1]
    fullname = os.path.join(path, fname)
    retry_cnt = 0

    while not (os.path.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError(
                "Download from {} failed. " "Retry limit reached".format(url)
            )

        logger.info("Downloading {} from {}".format(fname, url))

        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.info(
                "Downloading {} from {} failed {} times with exception {}".format(
                    fname, url, retry_cnt + 1, str(e)
                )
            )
            time.sleep(1)
            continue

        if req.status_code != 200:
            raise RuntimeError(
                "Downloading from {} failed with code "
                "{}!".format(url, req.status_code)
            )

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname


def _uncompress_file_zip(filepath):
    files = zipfile.ZipFile(filepath, "r")
    file_list = files.namelist()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)
        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    return uncompressed_path


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < -1:
        return True
    return False


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if "/" in file_path:
            file_path = file_path.replace("/", os.sep)
        elif "\\" in file_path:
            file_path = file_path.replace("\\", os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True


def _uncompress_file_tar(filepath, mode="r:*"):
    files = tarfile.open(filepath, mode)
    file_list = files.getnames()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)

        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def get_path_from_url(url, root_dir, md5sum=None, check_exist=True, decompress=True):
    """Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    from paddle.distributed import ParallelEnv

    assert is_url(url), "downloading from {} not a url".format(url)
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)
    # Mainly used to solve the problem of downloading data from different
    # machines in the case of multiple machines. Different nodes will download
    # data, and the same node will only download data once.
    rank_id_curr_node = int(os.environ.get("PADDLE_RANK_IN_NODE", 0))

    if os.path.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.info("Found {}".format(fullpath))
    else:
        if rank_id_curr_node == 0:
            fullpath = _download(url, root_dir, md5sum)
        else:
            while not os.path.exists(fullpath):
                time.sleep(1)

    if rank_id_curr_node == 0:
        if decompress and (
            tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath)
        ):
            fullpath = _decompress(fullpath)

    return fullpath


def get_weights_path_from_url(url, md5sum=None):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.

    Args:
        url (str): download url
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded weights.

    Examples:
        .. code-block:: python

            from paddle.utils.download import get_weights_path_from_url

            resnet18_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
            local_weight_path = get_weights_path_from_url(resnet18_pretrained_weight_url)

    """
    path = get_path_from_url(url, WEIGHTS_HOME, md5sum)
    return path


def load_dygraph_pretrain(
    model,
    pretrained_path,
    use_ssld=False,
    use_ssld_stage1_pretrained=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=False,
):
    if pretrained_path.startswith(("http://", "https://")):
        pretrained_path = _set_ssld_pretrained(
            pretrained_path,
            use_ssld=use_ssld,
            use_ssld_stage1_pretrained=use_ssld_stage1_pretrained,
        )
        if use_imagenet22k_pretrained:
            pretrained_path = pretrained_path.replace("_pretrained", "_22k_pretrained")
        if use_imagenet22kto1k_pretrained:
            pretrained_path = pretrained_path.replace(
                "_pretrained", "_22kto1k_pretrained"
            )
        pretrained_path = get_weights_path_from_url(pretrained_path)
    if not pretrained_path.endswith(".pdparams"):
        pretrained_path = pretrained_path + ".pdparams"
    if not os.path.exists(pretrained_path):
        raise ValueError(
            "Model pretrain path {} does not " "exists.".format(pretrained_path)
        )
    param_state_dict = paddle.load(pretrained_path)
    if isinstance(model, list):
        for m in model:
            if hasattr(m, "set_dict"):
                m.set_dict(param_state_dict)
    else:
        model.set_dict(param_state_dict)
    logger.info("Finish load pretrained model from {}".format(pretrained_path))
    return


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def pading_for_not_divisible(
    pixel_values, height, width, patch_size, format="BCHW", function="split"
):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if height % patch_size[0] == 0 and width % patch_size[1] == 0:
        return pixel_values, (0, 0, 0, 0, 0, 0, 0, 0)
    if function == "split":
        pading_width = patch_size[1] - width % patch_size[1]
        pading_height = patch_size[0] - height % patch_size[0]
    elif function == "merge":
        pading_width = width % 2
        pading_height = height % 2
    if format == "BCHW":
        pad_index = (0, 0, 0, 0, 0, pading_height, 0, pading_width)
    elif format == "BHWC":
        pad_index = (0, 0, 0, pading_height, 0, pading_width, 0, 0)
    else:
        assert "vaild format"

    return F.pad(pixel_values, pad_index), pad_index


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    x = windows.reshape(
        [-1, H // window_size, W // window_size, window_size, window_size, C]
    )
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])
    return x


class WindowAttention(nn.Layer):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_fused_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = self.create_parameter(
            shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
            default_initializer=zeros_,
        )
        self.add_parameter(
            "relative_position_bias_table", self.relative_position_bias_table
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww

        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2

        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

        self.use_fused_attn = use_fused_attn

    def eval(
        self,
    ):
        # this is used to re-param swin for model export
        relative_position_bias_table = self.relative_position_bias_table
        window_size = self.window_size
        index = self.relative_position_index.reshape([-1])

        relative_position_bias = paddle.index_select(
            relative_position_bias_table, index
        )
        relative_position_bias = relative_position_bias.reshape(
            [window_size[0] * window_size[1], window_size[0] * window_size[1], -1]
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            [2, 0, 1]
        )  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.unsqueeze(0)
        self.register_buffer("relative_position_bias", relative_position_bias)

    def get_relative_position_bias(self):
        if self.training or not hasattr(self, "relative_position_bias"):
            index = self.relative_position_index.reshape([-1])

            relative_position_bias = paddle.index_select(
                self.relative_position_bias_table, index
            )
            relative_position_bias = relative_position_bias.reshape(
                [
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1,
                ]
            )  # Wh*Ww,Wh*Ww,nH

            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1]
            )  # nH, Wh*Ww, Wh*Ww
            return relative_position_bias.unsqueeze(0)
        else:
            return self.relative_position_bias

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape([B_, N, 3, self.num_heads, C // self.num_heads])

        if self.use_fused_attn:
            qkv = qkv.transpose((2, 0, 1, 3, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_mask = self.get_relative_position_bias()
            if mask is not None:
                nW = mask.shape[0]
                mask = mask.reshape((1, nW, 1, N, N)).expand(
                    (B_ // nW, -1, self.num_heads, -1, -1)
                )
                attn_mask = attn_mask + mask.reshape((-1, self.num_heads, N, N))
            attn_mask = attn_mask.expand((B_, -1, -1, -1))
            attn = paddle.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            qkv = qkv.transpose((2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))
            attn = attn + self.get_relative_position_bias()

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.reshape(
                    [B_ // nW, nW, self.num_heads, N, N]
                ) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.reshape([B_, self.num_heads, N, N])
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            attn = paddle.mm(attn, v)
            attn = attn.transpose([0, 2, 1, 3])
        x = attn.reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return "dim={}, window_size={}, num_heads={}".format(
            self.dim, self.window_size, self.num_heads
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Layer):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_fused_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_fused_attn=use_fused_attn,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        H, W = self.input_resolution
        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for shifted window multihead self attention
            img_mask = paddle.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(
                (-1, self.window_size * self.window_size)
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self, x, input_dimensions):
        H, W = input_dimensions
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        x, pad_values = pading_for_not_divisible(x, H, W, self.window_size, "BHWC")
        _, height_pad, width_pad, _ = x.shape

        padding_state = pad_values[3] > 0 or pad_values[5] > 0  # change variable name
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size, C]
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # check did it need to calculate again
        attn_mask = self.get_attn_mask(height_pad, width_pad, x.dtype)

        attn_windows = self.attn(
            x_windows, mask=attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(
            attn_windows, self.window_size, height_pad, width_pad, C
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x

        if padding_state:
            x = x[:, :H, :W, :]
        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self):
        return "dim={}, input_resolution={}, num_heads={}, window_size={}, shift_size={}, mlp_ratio={}".format(
            self.dim,
            self.input_resolution,
            self.num_heads,
            self.window_size,
            self.shift_size,
            self.mlp_ratio,
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Layer):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, input_dimensions):
        """
        x: B, H*W, C
        """
        H, W = input_dimensions
        B, L, C = x.shape
        x = x.reshape([B, H, W, C])
        x, _ = pading_for_not_divisible(x, H, W, 2, "BHWC", function="merge")

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.reshape(x, x.shape)
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        # x = x.reshape([B, H // 2, 2, W // 2, 2, C])
        # x = x.transpose((0, 1, 3, 4, 2, 5))

        x = x.reshape([B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return "input_resolution={}, dim={}".format(self.input_resolution, self.dim)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Layer):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        use_fused_attn=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.LayerList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    use_fused_attn=use_fused_attn,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, input_dimensions):
        H, W = input_dimensions
        for blk in self.blocks:
            x = blk(x, input_dimensions)
        if self.downsample is not None:
            H, W = (H + 1) // 2, (W + 1) // 2
            x = self.downsample(x, input_dimensions)
        return x, (H, W)

    def extra_repr(self):
        return "dim={}, input_resolution={}, depth={}".format(
            self.dim, self.input_resolution, self.depth
        )

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if len(x) == 1:
            x = x[0]
        else:
            try:
                x = np.array(x)
            except Exception as e:
                print(f"len(x): {len(x)}, x: {x}")
                raise e
            x = x.reshape((x.shape[0], x.shape[2], x.shape[3], x.shape[4]))
            x = paddle.to_tensor(x, dtype=np.int16)

        if len(x.shape) == 3:
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

        B, C, H, W = x.shape

        x, _ = pading_for_not_divisible(x, H, W, self.patch_size, "BCHW")
        x = self.proj(x)
        _, _, height, width = x.shape
        output_dimensions = (height, width)
        x = x.flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, output_dimensions

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Layer):
    """Swin Transformer
        A PaddlePaddle impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        class_num=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes = class_num
        print(f"self.num_classes: {self.num_classes}")
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.in_channels = in_chans
        self.out_channels = embed_dim
        use_fused_attn = check_support_fused_op(kwargs.get("use_fused_attn", False))
        use_fused_linear = kwargs.get("use_fused_linear", False)
        global Linear
        Linear = paddle.incubate.nn.FusedLinear if use_fused_linear else nn.Linear

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = self.create_parameter(
                shape=(1, num_patches, embed_dim), default_initializer=zeros_
            )
            self.add_parameter("absolute_pos_embed", self.absolute_pos_embed)
            trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(
            0, drop_path_rate, sum(depths)
        ).tolist()  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_fused_attn=use_fused_attn,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1D(1)

        # self.head = (
        #     Linear(self.num_features, num_classes)
        #     if self.num_classes > 0
        #     else nn.Identity()
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, paddle.incubate.nn.FusedLinear)):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x, output_dimensions = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x, output_dimensions = layer(x, output_dimensions)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose([0, 2, 1]))  # B C 1
        x = paddle.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


def _load_pretrained(
    pretrained,
    model,
    model_url,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=False,
    **kwargs,
):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(
            model,
            model_url,
            use_ssld=use_ssld,
            use_imagenet22k_pretrained=use_imagenet22k_pretrained,
            use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
        )
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def SwinTransformer_tiny_patch4_window7_224(
    pretrained=True,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=False,
    **kwargs,
):
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.1
        class_num=kwargs["num_classes"],
        **kwargs,
    )

    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_tiny_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
    )
    return model


def SwinTransformer_small_patch4_window7_224(
    pretrained=True,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=False,
    **kwargs,
):
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs,
    )
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_small_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
    )
    return model


def SwinTransformer_base_patch4_window7_224(
    pretrained=False,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=False,
    **kwargs,
):
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs,
    )
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_base_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
    )
    return model


def SwinTransformer_base_patch4_window12_384(
    pretrained=False,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=False,
    **kwargs,
):
    model = SwinTransformer(
        img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.5,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs,
    )
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_base_patch4_window12_384"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
    )
    return model


def SwinTransformer_large_patch4_window7_224(
    pretrained=False,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=True,
    **kwargs,
):
    model = SwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        **kwargs,
    )
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_large_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
    )
    return model


def SwinTransformer_large_patch4_window12_384(
    pretrained=False,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=True,
    **kwargs,
):
    model = SwinTransformer(
        img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        **kwargs,
    )
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_large_patch4_window12_384"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
    )
    return model
