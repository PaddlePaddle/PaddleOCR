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

from ...device import get_cudnn_version
from ...fluid.framework import core, in_dygraph_mode
from ...static import Variable
from ...fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid import dygraph_utils
import numpy as np
from paddle import _C_ops

__all__ = []


def affine_grid(theta, out_shape, align_corners=True, name=None):
    """
    It generates a grid of (x,y) coordinates using the parameters of
    the affine transformation that correspond to a set of points where
    the input feature map should be sampled to produce the transformed
    output feature map.

    Args:
        theta (Tensor) - A tensor with shape [N, 2, 3]. It contains a batch of affine transform parameters.
                           The data type can be float32 or float64.
        out_shape (Tensor | list | tuple): The shape of target output with format [batch_size, channel, height, width].
                                             ``out_shape`` can be a Tensor or a list or tuple. The data
                                             type must be int32.
        align_corners(bool): Whether to align corners of target feature map and source feature map. Default: True.
        name(str|None): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A Tensor with shape [batch_size, H, W, 2] while 'H' and 'W' are the height and width of feature map in affine transformation. The data type is the same as `theta`.

    Raises:
        ValueError: If the type of arguments is not supported.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np
            # theta shape = [1, 2, 3]
            theta = np.array([[[-0.7, -0.4, 0.3],
                               [ 0.6,  0.5, 1.5]]]).astype("float32")
            theta_t = paddle.to_tensor(theta)
            y_t = F.affine_grid(
                    theta_t,
                    [1, 2, 3, 3],
                    align_corners=False)
            print(y_t)
            
            #[[[[ 1.0333333   0.76666665]
            #   [ 0.76666665  1.0999999 ]
            #   [ 0.5         1.4333333 ]]
            #
            #  [[ 0.5666667   1.1666666 ]
            #   [ 0.3         1.5       ]
            #   [ 0.03333333  1.8333334 ]]
            #
            #  [[ 0.10000002  1.5666667 ]
            #   [-0.16666666  1.9000001 ]
            #   [-0.43333334  2.2333333 ]]]]
    """
    if not isinstance(theta, Variable):
        raise ValueError("The theta should be a Tensor.")

    cudnn_version = get_cudnn_version()
    if cudnn_version is not None and cudnn_version >= 6000 and align_corners:
        use_cudnn = True
    else:
        use_cudnn = False
    if core.is_compiled_with_rocm():
        use_cudnn = False  # ROCM platform do not have MIOPEN kernel for affine_grid

    if not (isinstance(out_shape, list) or isinstance(out_shape, tuple) or \
            isinstance(out_shape, Variable)):
        raise ValueError("The out_shape should be a list, tuple or Tensor.")

    if in_dygraph_mode():
        _out_shape = out_shape.numpy().tolist() if isinstance(
            out_shape, Variable) else out_shape
        return _C_ops.affine_grid(theta, "output_shape", _out_shape,
                                  "align_corners", align_corners, "use_cudnn",
                                  use_cudnn)

    helper = LayerHelper('affine_grid')
    check_variable_and_dtype(theta, 'theta', ['float32', 'float64'],
                             'affine_grid')
    out = helper.create_variable_for_type_inference(theta.dtype)
    ipts = {'Theta': theta}
    attrs = {"align_corners": align_corners, "use_cudnn": use_cudnn}
    if isinstance(out_shape, Variable):
        ipts['OutputShape'] = out_shape
        check_variable_and_dtype(out_shape, 'out_shape', ['int32'],
                                 'affine_grid')
    else:
        attrs['output_shape'] = out_shape

    helper.append_op(
        type='affine_grid',
        inputs=ipts,
        outputs={'Output': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def grid_sample(x,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
                name=None):
    """
    This operation samples input X by using bilinear interpolation or
    nearest interpolation based on flow field grid, which is usually
    generated by :code:`affine_grid` . The grid of shape [N, H, W, 2]
    is the concatenation of (x, y) coordinates with shape [N, H, W] each,
    where x is indexing the 4th dimension (in width dimension) of input
    data x and y is indexing the 3rd dimension (in height dimension),
    finally results is the bilinear interpolation or nearest value of 4 nearest corner
    points. The output tensor shape will be [N, C, H, W].


    Step 1:

    Get (x, y) grid coordinates and scale to [0, H-1/W-1].

    .. code-block:: text

        grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1)
        grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

    Step 2:
    
    Indices input data X with grid (x, y) in each [H, W] area, and bilinear
    interpolate point value by 4 nearest points or nearest interpolate point value
    by nearest point.

    .. code-block:: text

        wn ------- y_n ------- en
        |           |           |
        |          d_n          |
        |           |           |
        x_w --d_w-- grid--d_e-- x_e
        |           |           |
        |          d_s          |
        |           |           |
        ws ------- y_s ------- wn

        For bilinear interpolation:
        x_w = floor(x)              // west side x coord
        x_e = x_w + 1               // east side x coord
        y_n = floor(y)              // north side y coord
        y_s = y_s + 1               // south side y coord
        d_w = grid_x - x_w          // distance to west side
        d_e = x_e - grid_x          // distance to east side
        d_n = grid_y - y_n          // distance to north side
        d_s = y_s - grid_y          // distance to south side
        wn = X[:, :, y_n, x_w]      // north-west point value
        en = X[:, :, y_n, x_e]      // north-east point value
        ws = X[:, :, y_s, x_w]      // south-east point value
        es = X[:, :, y_s, x_w]      // north-east point value

        output = wn * d_e * d_s + en * d_w * d_s
                + ws * d_e * d_n + es * d_w * d_n

    Args:
        x(Tensor): The input tensor, which is a 4-d tensor with shape
                     [N, C, H, W], N is the batch size, C is the channel
                     number, H and W is the feature height and width.
                     The data type is float32 or float64.
        grid(Tensor): Input grid tensor of shape [N, grid_H, grid_W, 2]. The
                        data type is float32 or float64.
        mode(str, optional): The interpolation method which can be 'bilinear' or 'nearest'.
                         Default: 'bilinear'.
        padding_mode(str, optional) The padding method used when source index
                   is out of input images. It can be 'zeros', 'reflection' and 'border'.
                   Default: zeros.
        align_corners(bool, optional): If `align_corners` is true, it will projects
                   -1 and 1 to the centers of the corner pixels. Otherwise, it will
                   projects -1 and 1 to the image edges.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor, The shape of output is [N, C, grid_H, grid_W] in which `grid_H` is the height of grid and `grid_W` is the width of grid. The data type is same as input tensor.

    Examples:

        .. code-block:: python
        
            import paddle
            import paddle.nn.functional as F
            import numpy as np
            
            # shape=[1, 1, 3, 3]
            x = np.array([[[[-0.6,  0.8, -0.5],
                            [-0.5,  0.2,  1.2],
                            [ 1.4,  0.3, -0.2]]]]).astype("float64")
            
            # grid shape = [1, 3, 4, 2]
            grid = np.array(
                         [[[[ 0.2,  0.3],
                            [-0.4, -0.3],
                            [-0.9,  0.3],
                            [-0.9, -0.6]],
                           [[ 0.4,  0.1],
                            [ 0.9, -0.8],
                            [ 0.4,  0.5],
                            [ 0.5, -0.2]],
                           [[ 0.1, -0.8],
                            [-0.3, -1. ],
                            [ 0.7,  0.4],
                            [ 0.2,  0.8]]]]).astype("float64")
            
            
            x = paddle.to_tensor(x)
            grid = paddle.to_tensor(grid)
            y_t = F.grid_sample(
                x,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True)
            print(y_t)
            
            # output shape = [1, 1, 3, 4]
            # [[[[ 0.34   0.016  0.086 -0.448]
            #    [ 0.55  -0.076  0.35   0.59 ]
            #    [ 0.596  0.38   0.52   0.24 ]]]]
    """

    _modes = ['bilinear', 'nearest']
    _padding_modes = ['zeros', 'reflection', 'border']
    if mode not in _modes:
        raise ValueError(
            "The mode of grid sample function should be in {}, but got: {}".
            format(_modes, mode))
    if padding_mode not in _padding_modes:
        raise ValueError(
            "The padding mode of grid sample function should be in {}, but got: {}".
            format(_padding_modes, padding_mode))

    if not isinstance(align_corners, bool):
        raise ValueError("The align corners should be bool, but got: {}".format(
            align_corners))

    cudnn_version = get_cudnn_version()
    use_cudnn = False
    if not core.is_compiled_with_rocm() and (
            cudnn_version is not None
    ) and align_corners and mode == 'bilinear' and padding_mode == 'zeros':
        use_cudnn = True
        # CUDNN always computes gradients for all inputs
        x.stop_gradient = False
        grid.stop_gradient = False

    if in_dygraph_mode():
        attrs = ('mode', mode, 'padding_mode', padding_mode, 'align_corners',
                 align_corners, 'use_cudnn', use_cudnn)
        out = getattr(_C_ops, 'grid_sampler')(x, grid, *attrs)
    else:
        helper = LayerHelper("grid_sample", **locals())
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'grid_sample')
        check_variable_and_dtype(grid, 'grid', ['float32', 'float64'],
                                 'grid_sample')
        ipts = {'X': x, 'Grid': grid}
        attrs = {
            'mode': mode,
            'padding_mode': padding_mode,
            'align_corners': align_corners,
            'use_cudnn': use_cudnn
        }
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='grid_sampler',
            inputs=ipts,
            attrs=attrs,
            outputs={'Output': out})
    return out


def pixel_shuffle(x, upscale_factor, data_format="NCHW", name=None):
    """
    This API implements pixel shuffle operation.
    See more details in :ref:`api_nn_vision_PixelShuffle` .
    Parameters:
        x(Tensor): 4-D tensor, the data type should be float32 or float64.
        upscale_factor(int): factor to increase spatial resolution.
        data_format (str): The data format of the input and output data. An optional string from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in the order of: [batch_size, input_channels, input_height, input_width].
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
    Returns:
        Out(tensor): Reshaped tensor according to the new dimension.
    Raises:
        ValueError: If the square of upscale_factor cannot divide the channels of input.
    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np
            x = np.random.randn(2, 9, 4, 4).astype(np.float32)
            x_var = paddle.to_tensor(x)
            out_var = F.pixel_shuffle(x_var, 3)
            out = out_var.numpy()
            # (2, 1, 12, 12)
    """
    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Attr(data_format) should be 'NCHW' or 'NHWC'."
                         "But recevie Attr(data_format): {} ".format(
                             data_format))

    if in_dygraph_mode():
        return _C_ops.pixel_shuffle(x, "upscale_factor", upscale_factor,
                                    "data_format", data_format)

    helper = LayerHelper("pixel_shuffle", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'pixel_shuffle')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="pixel_shuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"upscale_factor": upscale_factor,
               "data_format": data_format})
    return out
