#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import abc
import copy
import math
import numpy as np

import paddle

from . import utils
from ..cal_kl_threshold import cal_kl_threshold

__all__ = [
    'BaseQuantizer', 'AbsmaxQuantizer', 'PerChannelAbsmaxQuantizer',
    'KLQuantizer', 'HistQuantizer', 'SUPPORT_ACT_QUANTIZERS',
    'SUPPORT_WT_QUANTIZERS'
]


def abs_max_value(tensor):
    return float(paddle.max(paddle.abs(tensor)).numpy())


def merge_max_value(old, new):
    """
    Merge the max element one by one in two lists.
    """
    assert isinstance(old, list) and isinstance(new, list)
    if old != []:
        assert len(old) == len(new)
        for i in range(len(old)):
            assert type(old[i]) == type(new[i])
            if isinstance(old[i], list):
                new[i] = merge_max_value(old[i], new[i])
            else:
                new[i] = old[i] if new[i] < old[i] else new[i]
    return new


def combine_abs_max_and_hist(tensor, origin_max, origin_hist, bins,
                             upsample_bins):
    """
    """

    new_max = abs_max_value(tensor)

    if new_max == 0.0:
        return origin_max, origin_hist
    elif origin_max == 0.0:
        new_hist, _ = np.histogram(
            paddle.abs(tensor).numpy(), range=(0, new_max), bins=bins)
        new_hist = new_hist.astype(np.float32)
        return new_max, new_hist
    elif new_max <= origin_max:
        new_hist, _ = np.histogram(
            paddle.abs(tensor).numpy(), range=(0, origin_max), bins=bins)
        new_hist = new_hist.astype(np.float32)
        new_hist += origin_hist
        return origin_max, new_hist
    else:
        # bin_width = origin_max / (bins * upsample_bins) 
        #           = new_max / (bins * downsample_bins)
        bin_width = origin_max / (bins * upsample_bins)
        downsampe_bins = int(math.ceil(new_max / (bins * bin_width)))
        new_max = bins * bin_width * downsampe_bins

        upsampled_hist = np.repeat(origin_hist, upsample_bins)
        expanded_hist = np.zeros((bins * downsampe_bins), dtype=np.float32)
        expanded_hist[0:bins * upsample_bins] = upsampled_hist
        cumsumed_hist = np.cumsum(
            expanded_hist, dtype=np.float64)[downsampe_bins - 1::downsampe_bins]
        shift_cumsumed_hist = np.zeros((bins), dtype=np.float64)
        shift_cumsumed_hist[1:] = cumsumed_hist[0:-1]
        sampled_hist = (cumsumed_hist - shift_cumsumed_hist) / upsample_bins
        sampled_hist = sampled_hist.astype(np.float32)

        new_hist, _ = np.histogram(
            paddle.abs(tensor).numpy(), range=(0, new_max), bins=bins)
        new_hist = new_hist.astype(np.float32)
        new_hist += sampled_hist

        return new_max, new_hist


@six.add_metaclass(abc.ABCMeta)
class BaseQuantizer(object):
    """
    Base quantizer for activation and weight.
    """

    def __init__(self, quant_bits=8):
        super(BaseQuantizer, self).__init__()
        assert isinstance(quant_bits, int)
        assert quant_bits > 0 and quant_bits <= 16

        self.quant_bits = quant_bits

        self.abs_max_vals = []
        self.thresholds = []

    @abc.abstractmethod
    def sample_data(self, layer, tensors):
        pass

    @abc.abstractmethod
    def cal_thresholds(self):
        pass


class AbsmaxQuantizer(BaseQuantizer):
    """
    Per-tensor abs max quantizer.
    """

    def __init__(self, quant_bits=8):
        super(AbsmaxQuantizer, self).__init__(quant_bits)

    def sample_data(self, layer, tensors):
        assert isinstance(tensors, tuple)

        abs_max_vals = [abs_max_value(t) for t in tensors]
        self.abs_max_vals = merge_max_value(self.abs_max_vals, abs_max_vals)

    def cal_thresholds(self):
        self.thresholds = self.abs_max_vals


class PerChannelAbsmaxQuantizer(BaseQuantizer):
    """
    Per channel abs max quantizer.
    """

    def __init__(self, quant_bits=8):
        super(PerChannelAbsmaxQuantizer, self).__init__(quant_bits)

    def sample_data(self, layer, tensors):
        assert isinstance(layer, paddle.nn.Layer)
        assert isinstance(tensors, tuple)

        abs_max_vals_list = []
        for idx, tensor in enumerate(tensors):
            if isinstance(layer, tuple(utils.spec_channel_axis_layers)):
                abs_max_vals = [
                    abs_max_value(tensor[:, i]) for i in range(tensor.shape[1])
                ]
                abs_max_vals_list.append(abs_max_vals)
            else:
                abs_max_vals = [
                    abs_max_value(tensor[i]) for i in range(tensor.shape[0])
                ]
                abs_max_vals_list.append(abs_max_vals)

        self.abs_max_vals = merge_max_value(self.abs_max_vals,
                                            abs_max_vals_list)

    def cal_thresholds(self):
        self.thresholds = self.abs_max_vals


@six.add_metaclass(abc.ABCMeta)
class BaseHistQuantizer(BaseQuantizer):
    """
    """

    def __init__(self, quant_bits=8, bins=1024, upsample_bins=64):
        super(BaseHistQuantizer, self).__init__(quant_bits)
        self.bins = bins
        self.upsample_bins = upsample_bins

        self.hists = []

    def sample_data(self, layer, tensors):
        assert isinstance(tensors, tuple)

        if self.abs_max_vals == []:
            abs_max_vals = [abs_max_value(t) for t in tensors]
            self.abs_max_vals = abs_max_vals

            for idx, tensor in enumerate(tensors):
                if abs_max_vals[idx] == 0.0:
                    self.hists.append(None)
                else:
                    hist, _ = np.histogram(
                        paddle.abs(tensor).numpy(),
                        range=(0., abs_max_vals[idx]),
                        bins=self.bins)
                    hist = hist.astype(np.float32)
                    self.hists.append(hist)
        else:
            assert len(self.abs_max_vals) == len(tensors)
            assert len(self.hists) == len(tensors)

            for idx, tensor in enumerate(tensors):
                new_abs_max, new_hist = combine_abs_max_and_hist(
                    tensor, self.abs_max_vals[idx], self.hists[idx], self.bins,
                    self.upsample_bins)
                self.abs_max_vals[idx] = new_abs_max
                self.hists[idx] = new_hist

    @abc.abstractmethod
    def cal_thresholds(self):
        pass


class HistQuantizer(BaseHistQuantizer):
    """
    """

    def __init__(self,
                 quant_bits=8,
                 bins=1024,
                 upsample_bins=64,
                 hist_percent=0.99999):
        super(HistQuantizer, self).__init__(quant_bits, bins, upsample_bins)
        self.hist_percent = hist_percent

    def cal_thresholds(self):
        def _helper(abs_max, hist, percent):
            assert hist.ndim == 1 and percent < 1.0
            hist = hist / np.sum(hist, dtype=np.float64)
            cumsumed_hist = np.cumsum(hist)
            index = np.argwhere(cumsumed_hist >= percent)[0]
            return float((index - 0.5) * (abs_max / hist.shape[0]))

        for idx in range(len(self.hists)):
            if self.hists[idx] is None:
                self.thresholds.append(self.abs_max_vals[idx])
            else:
                threshold = _helper(self.abs_max_vals[idx], self.hists[idx],
                                    self.hist_percent)
                self.thresholds.append(threshold)


class KLQuantizer(BaseHistQuantizer):
    """
    """

    def __init__(self, quant_bits=8, bins=1024, upsample_bins=64):
        super(KLQuantizer, self).__init__(quant_bits, bins, upsample_bins)

    def cal_thresholds(self):
        for idx in range(len(self.hists)):
            if self.hists[idx] is None:
                self.thresholds.append(self.abs_max_vals[idx])
            else:
                hist = self.hists[idx]
                abs_max_val = self.abs_max_vals[idx]
                bin_width = abs_max_val / hist.shape[0]
                threshold = cal_kl_threshold(hist, bin_width, self.quant_bits)
                self.thresholds.append(threshold)


SUPPORT_ACT_QUANTIZERS = [AbsmaxQuantizer, HistQuantizer, KLQuantizer]
SUPPORT_WT_QUANTIZERS = [AbsmaxQuantizer, PerChannelAbsmaxQuantizer]
