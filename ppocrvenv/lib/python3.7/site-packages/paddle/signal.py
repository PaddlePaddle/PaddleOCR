# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional

import paddle

from .tensor.attribute import is_complex, is_floating_point
from .fft import fft_r2c, fft_c2r, fft_c2c
from .fluid.data_feeder import check_variable_and_dtype
from .fluid.framework import in_dygraph_mode
from .fluid.layer_helper import LayerHelper
from . import _C_ops

__all__ = [
    'stft',
    'istft',
]


def frame(x, frame_length, hop_length, axis=-1, name=None):
    """
    Slice the N-dimensional (where N >= 1) input into (overlapping) frames.

    Args:
        x (Tensor): The input data which is a N-dimensional (where N >= 1) Tensor
            with shape `[..., seq_length]` or `[seq_length, ...]`.
        frame_length (int): Length of the frame and `0 < frame_length <= x.shape[axis]`.
        hop_length (int): Number of steps to advance between adjacent frames
            and `0 < hop_length`. 
        axis (int, optional): Specify the axis to operate on the input Tensors. Its
            value should be 0(the first dimension) or -1(the last dimension). If not
            specified, the last axis is used by default. 

    Returns:
        The output frames tensor with shape `[..., frame_length, num_frames]` if `axis==-1`,
            otherwise `[num_frames, frame_length, ...]` where
        
            `num_framse = 1 + (x.shape[axis] - frame_length) // hop_length`

    Examples:

    .. code-block:: python

        import paddle
        from paddle.signal import frame
        
        # 1D
        x = paddle.arange(8)
        y0 = frame(x, frame_length=4, hop_length=2, axis=-1)  # [4, 3]
        # [[0, 2, 4],
        #  [1, 3, 5],
        #  [2, 4, 6],
        #  [3, 5, 7]]

        y1 = frame(x, frame_length=4, hop_length=2, axis=0)   # [3, 4]
        # [[0, 1, 2, 3],
        #  [2, 3, 4, 5],
        #  [4, 5, 6, 7]]

        # 2D
        x0 = paddle.arange(16).reshape([2, 8])
        y0 = frame(x0, frame_length=4, hop_length=2, axis=-1)  # [2, 4, 3]
        # [[[0, 2, 4],
        #   [1, 3, 5],
        #   [2, 4, 6],
        #   [3, 5, 7]],
        #
        #  [[8 , 10, 12],
        #   [9 , 11, 13],
        #   [10, 12, 14],
        #   [11, 13, 15]]]

        x1 = paddle.arange(16).reshape([8, 2])
        y1 = frame(x1, frame_length=4, hop_length=2, axis=0)   # [3, 4, 2]
        # [[[0 , 1 ],
        #   [2 , 3 ],
        #   [4 , 5 ],
        #   [6 , 7 ]],
        #
        #   [4 , 5 ],
        #   [6 , 7 ],
        #   [8 , 9 ],
        #   [10, 11]],
        #
        #   [8 , 9 ],
        #   [10, 11],
        #   [12, 13],
        #   [14, 15]]]

        # > 2D
        x0 = paddle.arange(32).reshape([2, 2, 8])
        y0 = frame(x0, frame_length=4, hop_length=2, axis=-1)  # [2, 2, 4, 3]

        x1 = paddle.arange(32).reshape([8, 2, 2])
        y1 = frame(x1, frame_length=4, hop_length=2, axis=0)   # [3, 4, 2, 2]
    """
    if axis not in [0, -1]:
        raise ValueError(f'Unexpected axis: {axis}. It should be 0 or -1.')

    if not isinstance(frame_length, int) or frame_length <= 0:
        raise ValueError(
            f'Unexpected frame_length: {frame_length}. It should be an positive integer.'
        )

    if not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError(
            f'Unexpected hop_length: {hop_length}. It should be an positive integer.'
        )

    if frame_length > x.shape[axis]:
        raise ValueError(
            f'Attribute frame_length should be less equal than sequence length, '
            f'but got ({frame_length}) > ({x.shape[axis]}).')

    op_type = 'frame'

    if in_dygraph_mode():
        attrs = ('frame_length', frame_length, 'hop_length', hop_length, 'axis',
                 axis)
        op = getattr(_C_ops, op_type)
        out = op(x, *attrs)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float16', 'float32',
                     'float64'], op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type=op_type,
            inputs={'X': x},
            attrs={
                'frame_length': frame_length,
                'hop_length': hop_length,
                'axis': axis
            },
            outputs={'Out': out})
    return out


def overlap_add(x, hop_length, axis=-1, name=None):
    """
    Reconstructs a tensor consisted of overlap added sequences from input frames.

    Args:
        x (Tensor): The input data which is a N-dimensional (where N >= 2) Tensor
            with shape `[..., frame_length, num_frames]` or
            `[num_frames, frame_length ...]`.
        hop_length (int): Number of steps to advance between adjacent frames and
            `0 < hop_length <= frame_length`. 
        axis (int, optional): Specify the axis to operate on the input Tensors. Its
            value should be 0(the first dimension) or -1(the last dimension). If not
            specified, the last axis is used by default. 

    Returns:
        The output frames tensor with shape `[..., seq_length]` if `axis==-1`,
            otherwise `[seq_length, ...]` where

            `seq_length = (n_frames - 1) * hop_length + frame_length`

    Examples:

    .. code-block:: python

        import paddle
        from paddle.signal import overlap_add
        
        # 2D
        x0 = paddle.arange(16).reshape([8, 2])
        # [[0 , 1 ],
        #  [2 , 3 ],
        #  [4 , 5 ],
        #  [6 , 7 ],
        #  [8 , 9 ],
        #  [10, 11],
        #  [12, 13],
        #  [14, 15]]
        y0 = overlap_add(x0, hop_length=2, axis=-1)  # [10]
        # [0 , 2 , 5 , 9 , 13, 17, 21, 25, 13, 15]

        x1 = paddle.arange(16).reshape([2, 8])
        # [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ],
        #  [8 , 9 , 10, 11, 12, 13, 14, 15]]
        y1 = overlap_add(x1, hop_length=2, axis=0)   # [10]
        # [0 , 1 , 10, 12, 14, 16, 18, 20, 14, 15]

        # > 2D
        x0 = paddle.arange(32).reshape([2, 1, 8, 2])
        y0 = overlap_add(x0, hop_length=2, axis=-1)  # [2, 1, 10]

        x1 = paddle.arange(32).reshape([2, 8, 1, 2])
        y1 = overlap_add(x1, hop_length=2, axis=0)   # [10, 1, 2] 
    """
    if axis not in [0, -1]:
        raise ValueError(f'Unexpected axis: {axis}. It should be 0 or -1.')

    if not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError(
            f'Unexpected hop_length: {hop_length}. It should be an positive integer.'
        )

    op_type = 'overlap_add'

    if in_dygraph_mode():
        attrs = ('hop_length', hop_length, 'axis', axis)
        op = getattr(_C_ops, op_type)
        out = op(x, *attrs)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float16', 'float32',
                     'float64'], op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type=op_type,
            inputs={'X': x},
            attrs={'hop_length': hop_length,
                   'axis': axis},
            outputs={'Out': out})
    return out


def stft(x,
         n_fft,
         hop_length=None,
         win_length=None,
         window=None,
         center=True,
         pad_mode='reflect',
         normalized=False,
         onesided=True,
         name=None):
    """
    Short-time Fourier transform (STFT).

    The STFT computes the discrete Fourier transforms (DFT) of short overlapping
    windows of the input using this formula:
    
    .. math::
        X_t[\omega] = \sum_{n = 0}^{N-1}%
                      \text{window}[n]\ x[t \times H + n]\ %
                      e^{-{2 \pi j \omega n}/{N}}
    
    Where:
    - :math:`t`: The :math:`t`-th input window.
    - :math:`\omega`: Frequency :math:`0 \leq \omega < \text{n\_fft}` for `onesided=False`,
        or :math:`0 \leq \omega < \lfloor \text{n\_fft} / 2 \rfloor + 1` for `onesided=True`. 
    - :math:`N`: Value of `n_fft`.
    - :math:`H`: Value of `hop_length`.  
    
    Args:
        x (Tensor): The input data which is a 1-dimensional or 2-dimensional Tensor with
            shape `[..., seq_length]`. It can be a real-valued or a complex Tensor.
        n_fft (int): The number of input samples to perform Fourier transform.
        hop_length (int, optional): Number of steps to advance between adjacent windows
            and `0 < hop_length`. Default: `None`(treated as equal to `n_fft//4`)
        win_length (int, optional): The size of window. Default: `None`(treated as equal
            to `n_fft`)
        window (Tensor, optional): A 1-dimensional tensor of size `win_length`. It will
            be center padded to length `n_fft` if `win_length < n_fft`. Default: `None`(
            treated as a rectangle window with value equal to 1 of size `win_length`).
        center (bool, optional): Whether to pad `x` to make that the
            :math:`t \times hop\_length` at the center of :math:`t`-th frame. Default: `True`.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. See
            `paddle.nn.functional.pad` for all padding options. Default: `"reflect"`
        normalized (bool, optional): Control whether to scale the output by `1/sqrt(n_fft)`.
            Default: `False`
        onesided (bool, optional): Control whether to return half of the Fourier transform
            output that satisfies the conjugate symmetry condition when input is a real-valued
            tensor. It can not be `True` if input is a complex tensor. Default: `True`
        name (str, optional): The default value is None. Normally there is no need for user
            to set this property. For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        The complex STFT output tensor with shape `[..., n_fft//2 + 1, num_frames]`(
            real-valued input and `onesided` is `True`) or `[..., n_fft, num_frames]`(
            `onesided` is `False`)
    
    Examples:
        .. code-block:: python
    
            import paddle
            from paddle.signal import stft
    
            # real-valued input
            x = paddle.randn([8, 48000], dtype=paddle.float64)
            y1 = stft(x, n_fft=512)  # [8, 257, 376]
            y2 = stft(x, n_fft=512, onesided=False)  # [8, 512, 376]
    
            # complex input
            x = paddle.randn([8, 48000], dtype=paddle.float64) + \
                    paddle.randn([8, 48000], dtype=paddle.float64)*1j  # [8, 48000] complex128
            y1 = stft(x, n_fft=512, center=False, onesided=False)  # [8, 512, 372]
    """
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'complex64', 'complex128'],
        'stft')

    x_rank = len(x.shape)
    assert x_rank in [1, 2], \
        f'x should be a 1D or 2D real tensor, but got rank of x is {x_rank}'

    if x_rank == 1:  # (batch, seq_length)
        x = x.unsqueeze(0)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    assert hop_length > 0, \
        f'hop_length should be > 0, but got {hop_length}.'

    if win_length is None:
        win_length = n_fft

    assert 0 < n_fft <= x.shape[-1], \
        f'n_fft should be in (0, seq_length({x.shape[-1]})], but got {n_fft}.'

    assert 0 < win_length <= n_fft, \
        f'win_length should be in (0, n_fft({n_fft})], but got {win_length}.'

    if window is not None:
        assert len(window.shape) == 1 and len(window) == win_length, \
            f'expected a 1D window tensor of size equal to win_length({win_length}), but got window with shape {window.shape}.'
    else:
        window = paddle.ones(shape=(win_length, ), dtype=x.dtype)

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = paddle.nn.functional.pad(window,
                                          pad=[pad_left, pad_right],
                                          mode='constant')

    if center:
        assert pad_mode in ['constant', 'reflect'], \
            'pad_mode should be "reflect" or "constant", but got "{}".'.format(pad_mode)

        pad_length = n_fft // 2
        # FIXME: Input `x` can be a complex tensor but pad does not supprt complex input.
        x = paddle.nn.functional.pad(x.unsqueeze(-1),
                                     pad=[pad_length, pad_length],
                                     mode=pad_mode,
                                     data_format="NLC").squeeze(-1)

    x_frames = frame(x=x, frame_length=n_fft, hop_length=hop_length, axis=-1)
    x_frames = x_frames.transpose(
        perm=[0, 2,
              1])  # switch n_fft to last dim, egs: (batch, num_frames, n_fft)
    x_frames = x_frames * window

    norm = 'ortho' if normalized else 'backward'
    if is_complex(x_frames):
        assert not onesided, \
            'onesided should be False when input or window is a complex Tensor.'

    if not is_complex(x):
        out = fft_r2c(
            x=x_frames,
            n=None,
            axis=-1,
            norm=norm,
            forward=True,
            onesided=onesided,
            name=name)
    else:
        out = fft_c2c(
            x=x_frames, n=None, axis=-1, norm=norm, forward=True, name=name)

    out = out.transpose(perm=[0, 2, 1])  # (batch, n_fft, num_frames)

    if x_rank == 1:
        out.squeeze_(0)

    return out


def istft(x,
          n_fft,
          hop_length=None,
          win_length=None,
          window=None,
          center=True,
          normalized=False,
          onesided=True,
          length=None,
          return_complex=False,
          name=None):
    """
    Inverse short-time Fourier transform (ISTFT).

    Reconstruct time-domain signal from the giving complex input and window tensor when
        nonzero overlap-add (NOLA) condition is met: 

    .. math::
        \sum_{t = -\infty}^{\infty}%
            \text{window}^2[n - t \times H]\ \neq \ 0, \ \text{for } all \ n

    Where:
    - :math:`t`: The :math:`t`-th input window.
    - :math:`N`: Value of `n_fft`.
    - :math:`H`: Value of `hop_length`.

    Result of `istft` expected to be the inverse of `paddle.signal.stft`, but it is
        not guaranteed to reconstruct a exactly realizible time-domain signal from a STFT
        complex tensor which has been modified (via masking or otherwise). Therefore, `istft`
        gives the [Griffin-Lim optimal estimate](https://ieeexplore.ieee.org/document/1164317)
        (optimal in a least-squares sense) for the corresponding signal.

    Args:
        x (Tensor): The input data which is a 2-dimensional or 3-dimensional **complesx**
            Tensor with shape `[..., n_fft, num_frames]`. 
        n_fft (int): The size of Fourier transform.
        hop_length (int, optional): Number of steps to advance between adjacent windows
            from time-domain signal and `0 < hop_length < win_length`. Default: `None`(
            treated as equal to `n_fft//4`)
        win_length (int, optional): The size of window. Default: `None`(treated as equal
            to `n_fft`)
        window (Tensor, optional): A 1-dimensional tensor of size `win_length`. It will
            be center padded to length `n_fft` if `win_length < n_fft`. It should be a
            real-valued tensor if `return_complex` is False. Default: `None`(treated as
            a rectangle window with value equal to 1 of size `win_length`).
        center (bool, optional): It means that whether the time-domain signal has been
            center padded. Default: `True`.
        normalized (bool, optional): Control whether to scale the output by `1/sqrt(n_fft)`.
            Default: `False`
        onesided (bool, optional): It means that whether the input STFT tensor is a half
            of the conjugate symmetry STFT tensor transformed from a real-valued signal
            and `istft` will return a real-valued tensor when it is set to `True`.
            Default: `True`.
        length (int, optional): Specify the length of time-domain signal. Default: `None`(
            treated as the whole length of signal). 
        return_complex (bool, optional): It means that whether the time-domain signal is
            real-valued. If `return_complex` is set to `True`, `onesided` should be set to
            `False` cause the output is complex. 
        name (str, optional): The default value is None. Normally there is no need for user
            to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A tensor of least squares estimation of the reconstructed signal(s) with shape
            `[..., seq_length]`

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.signal import stft, istft

            paddle.seed(0)

            # STFT
            x = paddle.randn([8, 48000], dtype=paddle.float64)
            y = stft(x, n_fft=512)  # [8, 257, 376]

            # ISTFT
            x_ = istft(y, n_fft=512)  # [8, 48000]

            np.allclose(x, x_)  # True
    """
    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], 'istft')

    x_rank = len(x.shape)
    assert x_rank in [2, 3], \
        'x should be a 2D or 3D complex tensor, but got rank of x is {}'.format(x_rank)

    if x_rank == 2:  # (batch, n_fft, n_frames)
        x = x.unsqueeze(0)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    if win_length is None:
        win_length = n_fft

    # Assure no gaps between frames.
    assert 0 < hop_length <= win_length, \
        'hop_length should be in (0, win_length({})], but got {}.'.format(win_length, hop_length)

    assert 0 < win_length <= n_fft, \
        'win_length should be in (0, n_fft({})], but got {}.'.format(n_fft, win_length)

    n_frames = x.shape[-1]
    fft_size = x.shape[-2]

    if onesided:
        assert (fft_size == n_fft // 2 + 1), \
            'fft_size should be equal to n_fft // 2 + 1({}) when onesided is True, but got {}.'.format(n_fft // 2 + 1, fft_size)
    else:
        assert (fft_size == n_fft), \
            'fft_size should be equal to n_fft({}) when onesided is False, but got {}.'.format(n_fft, fft_size)

    if window is not None:
        assert len(window.shape) == 1 and len(window) == win_length, \
            'expected a 1D window tensor of size equal to win_length({}), but got window with shape {}.'.format(win_length, window.shape)
    else:
        window = paddle.ones(shape=(win_length, ))

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        # FIXME: Input `window` can be a complex tensor but pad does not supprt complex input.
        window = paddle.nn.functional.pad(window,
                                          pad=[pad_left, pad_right],
                                          mode='constant')

    x = x.transpose(
        perm=[0, 2,
              1])  # switch n_fft to last dim, egs: (batch, num_frames, n_fft)
    norm = 'ortho' if normalized else 'backward'

    if return_complex:
        assert not onesided, \
            'onesided should be False when input(output of istft) or window is a complex Tensor.'

        out = fft_c2c(x=x, n=None, axis=-1, norm=norm, forward=False, name=None)
    else:
        assert not is_complex(window), \
            'Data type of window should not be complex when return_complex is False.'

        if onesided is False:
            x = x[:, :, :n_fft // 2 + 1]
        out = fft_c2r(x=x, n=None, axis=-1, norm=norm, forward=False, name=None)

    out = overlap_add(
        x=(out * window).transpose(
            perm=[0, 2, 1]),  # (batch, n_fft, num_frames)
        hop_length=hop_length,
        axis=-1)  # (batch, seq_length)

    window_envelop = overlap_add(
        x=paddle.tile(
            x=window * window, repeat_times=[n_frames, 1]).transpose(
                perm=[1, 0]),  # (n_fft, num_frames)
        hop_length=hop_length,
        axis=-1)  # (seq_length, )

    if length is None:
        if center:
            out = out[:, (n_fft // 2):-(n_fft // 2)]
            window_envelop = window_envelop[(n_fft // 2):-(n_fft // 2)]
    else:
        if center:
            start = n_fft // 2
        else:
            start = 0

        out = out[:, start:start + length]
        window_envelop = window_envelop[start:start + length]

    # Check whether the Nonzero Overlap Add (NOLA) constraint is met.
    if window_envelop.abs().min().item() < 1e-11:
        raise ValueError(
            'Abort istft because Nonzero Overlap Add (NOLA) condition failed. For more information about NOLA constraint please see `scipy.signal.check_NOLA`(https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html).'
        )

    out = out / window_envelop

    if x_rank == 2:
        out.squeeze_(0)

    return out
