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

import warnings
import paddle
from ...fluid.framework import in_dygraph_mode, default_main_program
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.tensor import fill_constant
from ...tensor import concat
from ...tensor.creation import zeros
from paddle.static import Variable
from ...fluid.layers import core
from ...fluid import dygraph_utils
# TODO: define the common functions to build a neural network  
from ...fluid.layers import unfold  # noqa: F401
from ...tensor.manipulation import squeeze
from ...tensor.manipulation import unsqueeze
from ...tensor import clip
from ...tensor import sum
from ...tensor import sqrt
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
from ...fluid.framework import in_dygraph_mode, _varbase_creator

from ...fluid.framework import in_dygraph_mode
from ...fluid import core, dygraph_utils
from ...fluid import core, layers
from ...fluid.data_feeder import check_variable_and_dtype
from paddle import _C_ops

__all__ = []


def interpolate(x,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=False,
                align_mode=0,
                data_format='NCHW',
                name=None):
    """

    This op resizes a batch of images.
    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or 4-D (num_batches, channels, in_h, in_w), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    Where in_w is width of the input tensor, in_h is the height of the input tensor,
    in_d is the depth of the intput tensor.
    and the resizing only applies on the three dimensions(depth, height and width).

    Supporting resample methods:
        'linear' : Linear interpolation
        'bilinear' : Bilinear interpolation
        'trilinear' : Trilinear interpolation
        'nearest' : Nearest neighbor interpolation
        'bicubic' : Bicubic interpolation
        'area': Area interpolation

    Linear interpolation is the method of using a line connecting two known quantities 
    to determine the value of an unknown quantity between the two known quantities. 
    
    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.
    align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Area interpolation is to perform area interpolation
    in both the 3rd dimension(in height direction) , the 4th dimension(in width
    direction) and the 5th dimension(in depth direction) on input tensor. Set to 
    area will directly call `paddle.nn.functional.adaptive_avg_pool1d` or 
    `paddle.nn.functional.adaptive_avg_pool2d` or `paddle.nn.functional.adaptive_avg_pool3d`.

    Example:

    .. code-block:: text

        For scale_factor:
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            else:
              scale_factor = float(in_size/out_size)

        Linear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = W_{in} * scale_{factor}
        
        Nearest neighbor interpolation:

              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})

        Bilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Bicubic interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    For details of linear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Linear_interpolation.
    
    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.
    
    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.
    
    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.
    
    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation
    
    Parameters:
        x (Tensor): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w) 
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor. 
             Default: None. If a list/tuple, each element can be an integer or a Tensor of shape: [1].
             If a Tensor, its dimensions size should be a 1.
        scale_factor (float|Tensor|list|tuple|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        mode (str): The resample method. It supports 'linear', 'area', 'nearest', 'bilinear',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.This only has an effect when 'linear', 'bilinear', 'bicubic' or 'trilinear'.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_indx+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`,  `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 3-D Tensor of the shape (num_batches, channels, out_w) or (num_batches, out_w, channels),
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).
    Raises:
        TypeError: size should be a list or tuple or Tensor.
        ValueError: The 'mode' of image_resize can only be 'linear', 'bilinear',
                    'trilinear', 'bicubic', 'area' or 'nearest' currently.
        ValueError: 'linear' only support 3-D tensor.
        ValueError: 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        ValueError: 'trilinear' only support 5-D tensor.
        ValueError: One of size and scale_factor must not be None.
        ValueError: size length should be 1 for input 3-D tensor.
        ValueError: size length should be 2 for input 4-D tensor.
        ValueError: size length should be 3 for input 5-D tensor.
        ValueError: scale_factor should be greater than zero.
        TypeError: align_corners should be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.

    Examples:
        .. code-block:: python

	        import paddle
	        import numpy as np
            import paddle.nn.functional as F
            
            # given out size
            input_data = np.random.rand(2,3,6,10).astype("float32")
            x = paddle.to_tensor(input_data)
            output_1 = F.interpolate(x=x, size=[12,12])
    	    print(output_1.shape)
	        # [2L, 3L, 12L, 12L]
            
            # given scale
            output_2 = F.interpolate(x=x, scale_factor=[2,1])
            print(output_2.shape)
            # [2L, 3L, 12L, 10L]
            
            # bilinear interp
            output_3 = F.interpolate(x=x, scale_factor=[2,1], mode="bilinear")
            print(output_2.shape)
            # [2L, 3L, 12L, 10L]
    """
    data_format = data_format.upper()
    resample = mode.upper()
    resample_type = mode.lower()

    resample_methods = [
        'LINEAR',
        'BILINEAR',
        'TRILINEAR',
        'NEAREST',
        'BICUBIC',
        'AREA',
    ]
    if resample not in resample_methods:
        raise ValueError(
            "The 'resample' of image_resize can only be 'area', 'linear', 'bilinear', 'trilinear', "
            " 'bicubic' or 'nearest' currently.")

    if resample in ['LINEAR'] and len(x.shape) != 3:
        raise ValueError("'linear' only support 3-D tensor.")

    if resample in ['BILINEAR', 'NEAREST', 'BICUBIC'] and len(x.shape) != 4:
        raise ValueError(
            "'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.")
    if resample == 'TRILINEAR' and len(x.shape) != 5:
        raise ValueError("'trilinear'only support 5-D tensor.")

    if size is None and scale_factor is None:
        raise ValueError("One of size and scale_factor must not be None.")

    if not isinstance(align_corners, bool):
        raise TypeError("Attr align_corners should be a bool value")

    if align_mode != 0 and align_mode != 1:
        raise ValueError("align_mode can only be 0 or 1")
    if align_corners != 0 and resample == 'NEAREST':
        raise ValueError(
            "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
        )

    if resample == 'AREA':
        if isinstance(size, list) or isinstance(size, tuple):
            if len(size) == 0:
                raise ValueError("output size can not be empty")
        if len(x.shape) == 3:
            return paddle.nn.functional.adaptive_avg_pool1d(x, size)
        elif len(x.shape) == 4:
            return paddle.nn.functional.adaptive_avg_pool2d(x, size)
        elif len(x.shape) == 5:
            return paddle.nn.functional.adaptive_avg_pool3d(x, size)

    helper = LayerHelper('{}_interp_v2'.format(resample_type), **locals())
    dtype = helper.input_dtype(input_param_name='x')
    if len(x.shape) == 3 and data_format not in ['NCW', 'NWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCW` or `NWC` supported for 3-D input.")
    elif len(x.shape) == 4 and data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCHW` or `NHWC` supported for 4-D input.")
    elif len(x.shape) == 5 and data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCDHW` or `NDHWC` supported for 5-D input.")

    def _is_list_or_turple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if data_format == 'NCHW' or data_format == 'NCDHW' or data_format == 'NCW':
        data_layout = 'NCHW'
    if data_format == 'NHWC' or data_format == 'NDHWC' or data_format == 'NWC':
        data_layout = 'NHWC'

    if resample == 'NEAREST':
        align_corners = False

    inputs = {"X": x}
    attrs = {
        "out_d": -1,
        "out_h": -1,
        "out_w": -1,
        "interp_method": resample_type,
        "align_corners": align_corners,
        "align_mode": align_mode,
        "data_layout": data_layout
    }

    out_shape = size
    scale = scale_factor

    if out_shape is not None and scale is not None:
        raise ValueError("Only one of size or scale_factor should be defined.")
    if out_shape is not None:
        if isinstance(out_shape, Variable) and not in_dygraph_mode():
            out_shape.stop_gradient = True
            inputs['OutSize'] = out_shape
        else:
            if in_dygraph_mode():
                if isinstance(out_shape, Variable):
                    out_shape = list(out_shape.numpy())
                for i, dim in enumerate(out_shape):
                    if isinstance(dim, Variable):
                        out_shape[i] = dim.numpy()[0]
            if not (_is_list_or_turple_(out_shape)):
                raise TypeError("size should be a list or tuple or Variable.")
            # Validate the shape
            contain_var = False
            for dim_idx, dim_size in enumerate(out_shape):
                if isinstance(dim_size, Variable):
                    contain_var = True
                    continue
                assert dim_size > 0, (
                    "Each dimension size given in out_shape must be greater than 0."
                )

            if contain_var:
                new_size_tensor = []
                size_list = []
                for dim in out_shape:
                    if isinstance(dim, Variable):
                        dim.stop_gradient = True
                        new_size_tensor.append(dim)
                        size_list.append(-1)
                    else:
                        assert (isinstance(dim, int))
                        temp_out = helper.create_variable_for_type_inference(
                            'int32')
                        fill_constant(
                            [1], 'int32', dim, force_cpu=True, out=temp_out)
                        new_size_tensor.append(temp_out)
                        size_list.append(dim)
                inputs['SizeTensor'] = new_size_tensor

            if len(x.shape) == 3:
                if len(out_shape) != 1:
                    raise ValueError(
                        "size length should be 2 for input 3-D tensor")
                if contain_var:
                    attrs['out_w'] = size_list[0]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_w'] = out_shape[0]
            if len(x.shape) == 4:
                if len(out_shape) != 2:
                    raise ValueError("size length should be 2 for "
                                     "input 4-D tensor.")
                if contain_var:
                    attrs['out_h'] = size_list[0]
                    attrs['out_w'] = size_list[1]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_h'] = out_shape[0]
                    attrs['out_w'] = out_shape[1]
            if len(x.shape) == 5:
                if len(out_shape) != 3:
                    raise ValueError("size length should be 3 for "
                                     "input 5-D tensor.")
                if contain_var:
                    attrs['out_d'] = size_list[0]
                    attrs['out_h'] = size_list[1]
                    attrs['out_w'] = size_list[2]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_d'] = out_shape[0]
                    attrs['out_h'] = out_shape[1]
                    attrs['out_w'] = out_shape[2]

    else:
        if in_dygraph_mode() and isinstance(scale, Variable):
            scale = list(scale.numpy())
        if isinstance(scale, Variable):
            scale.stop_gradient = True
            inputs["Scale"] = scale
        elif isinstance(scale, float) or isinstance(scale, int):
            if scale <= 0:
                raise ValueError("Attr(scale) should be greater than zero.")
            scale_list = []
            for i in range(len(x.shape) - 2):
                scale_list.append(scale)
            attrs['scale'] = list(map(float, scale_list))
        elif isinstance(scale, list) or isinstance(scale, tuple):
            if len(scale) != len(x.shape) - 2:
                raise ValueError("scale_shape length should be {} for "
                                 "input {}-D tensor.".format(
                                     len(x.shape) - 2, len(x.shape)))
            for value in scale:
                if value <= 0:
                    raise ValueError("Attr(scale) should be greater than zero.")
            attrs['scale'] = list(map(float, scale))
        else:
            raise TypeError(
                "Attr(scale)'s type should be float, int, list, tuple, or Tensor."
            )

    if in_dygraph_mode():
        attr_list = []
        for k, v in attrs.items():
            attr_list.append(k)
            attr_list.append(v)
        dy_attr = tuple(attr_list)

        if resample_type == "linear":
            out = _C_ops.linear_interp_v2(x, *dy_attr)
        elif resample_type == "bilinear":
            out = _C_ops.bilinear_interp_v2(x, *dy_attr)
        elif resample_type == "trilinear":
            out = _C_ops.trilinear_interp_v2(x, *dy_attr)
        elif resample_type == "nearest":
            out = _C_ops.nearest_interp_v2(x, *dy_attr)
        elif resample_type == "bicubic":
            out = _C_ops.bicubic_interp_v2(x, *dy_attr)
        return out
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='{}_interp_v2'.format(resample_type),
        inputs=inputs,
        outputs={"Out": out},
        attrs=attrs)
    return out


def upsample(x,
             size=None,
             scale_factor=None,
             mode='nearest',
             align_corners=False,
             align_mode=0,
             data_format='NCHW',
             name=None):
    """
    This op resizes a batch of images.

    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or 4-D (num_batches, channels, in_h, in_w), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    Where in_w is width of the input tensor, in_h is the height of the input tensor,
    in_d is the depth of the intput tensor.
    and the resizing only applies on the three dimensions(depth, height and width).

    Supporting resample methods:
        'linear' : Linear interpolation
        'bilinear' : Bilinear interpolation
        'trilinear' : Trilinear interpolation
        'nearest' : Nearest neighbor interpolation
        'bicubic' : Bicubic interpolation
    Linear interpolation is the method of using a line connecting two known quantities 
    to determine the value of an unknown quantity between the two known quantities. 
    
    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.
    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.
    
    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.

    The linear interpolation is performed on three directions.
    align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Area interpolation is to perform area interpolation
    in both the 3rd dimension(in height direction) , the 4th dimension(in width
    direction) and the 5th dimension(in depth direction) on input tensor. Set to
    area will directly call `paddle.nn.functional.adaptive_avg_pool1d` or
    `paddle.nn.functional.adaptive_avg_pool2d` or `paddle.nn.functional.adaptive_avg_pool3d`.

    Example:
    .. code-block:: text
        For scale_factor:
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            else:
              scale_factor = float(in_size/out_size)
        Linear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = W_{in} * scale_{factor}
        Nearest neighbor interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})
          else:
              align_corners = True
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})
        
        Bilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}
        Bicubic interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}
        Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}
    https://en.wikipedia.org/wiki/Linear_interpolation.
    For details of linear interpolation, please refer to Wikipedia:
    
    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.
    
    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.
    
    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation
    
    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.
    
    Parameters:
        x (Tensor): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w) 
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor. 
             Default: None. If a list/tuple, each element can be an integer or a Tensor of shape: [1].
             If a Tensor , its dimensions size should be a 1.
        scale_factor (float|Tensor|list|tuple|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.Has to match input size if 
             it is either a list or a tuple or a Tensor.
             Default: None.
        mode (str): The resample method. It supports 'linear', 'nearest', 'bilinear',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_indx+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 3-D Tensor of the shape (num_batches, channels, out_w) or (num_batches, out_w, channels),
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).
    Raises:
        TypeError: size should be a list or tuple or Tensor.
        ValueError: The 'mode' of image_resize can only be 'linear', 'bilinear',
                    'trilinear', 'bicubic', or 'nearest' currently.
        ValueError: 'linear' only support 3-D tensor.
        ValueError: 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        ValueError: 'trilinear' only support 5-D tensor.
        ValueError: One of size and scale_factor must not be None.
        ValueError: size length should be 1 for input 3-D tensor.
        ValueError: size length should be 2 for input 4-D tensor.
        ValueError: size length should be 3 for input 5-D tensor.
        ValueError: scale_factor should be greater than zero.
        TypeError: align_corners should be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.
        Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            import paddle.nn.functional as F

            input_data = np.random.rand(2,3,6,10).astype("float32")
            input = paddle.to_tensor(input_data)
            output = F.upsample(x=input, size=[12,12])
            print(output.shape)
            # [2L, 3L, 12L, 12L]

    """
    return interpolate(x, size, scale_factor, mode, align_corners, align_mode,
                       data_format)


def bilinear(x1, x2, weight, bias=None, name=None):
    """

    This layer performs bilinear on two inputs.
    See :ref:`api_nn_Bilinear` for details and output shape.

    Parameters:
       x1 (Tensor): the first input tensor, it's data type should be float32, float64.
       x2 (Tensor): the second input tensor, it's data type should be float32, float64.
       weight (Parameter): The learnable weights of this layer, shape is [out_features, in1_features, in2_features].
       bias (Parameter, optional): The learnable bias(Bias) of this layer, shape is [1, out_features]. If it is set to None, no bias will be added to the output units. The default value is None.
       name (str, optional): The default value is None. Normally there is no need for user
           to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
       Tensor: A 2-D Tensor of shape [batch_size, out_features].

    Examples:
       .. code-block:: python

        import paddle
        import numpy
        import paddle.nn.functional as F

        x1 = numpy.random.random((5, 5)).astype('float32')
        x2 = numpy.random.random((5, 4)).astype('float32')
        w = numpy.random.random((1000, 5, 4)).astype('float32')
        b = numpy.random.random((1, 1000)).astype('float32')

        result = F.bilinear(paddle.to_tensor(x1), paddle.to_tensor(x2), paddle.to_tensor(w), paddle.to_tensor(b))           # result shape [5, 1000]

    """

    if in_dygraph_mode():
        return _C_ops.bilinear_tensor_product(x1, x2, weight, bias)

    check_variable_and_dtype(x1, 'x1', ['float32', 'float64'], 'bilinear')
    check_variable_and_dtype(x2, 'x2', ['float32', 'float64'], 'bilinear')

    inputs = {"X": x1, "Y": x2, "Weight": weight}
    if bias is not None:
        inputs["Bias"] = bias

    helper = LayerHelper("bilinear", **locals())
    out = helper.create_variable_for_type_inference(dtype=x1.dtype)

    helper.append_op(
        type="bilinear_tensor_product", inputs=inputs, outputs={"Out": out})

    return out


def dropout(x,
            p=0.5,
            axis=None,
            training=True,
            mode="upscale_in_train",
            name=None):
    """
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training. The dropout operator randomly sets the
    outputs of some units to zero, while upscale others according to the given
    dropout probability.

    Args:
        x (Tensor): The input tensor. The data type is float32 or float64.
        p (float|int): Probability of setting units to zero. Default 0.5.
        axis (int|list|tuple): The axis along which the dropout is performed. Default None.
        training (bool): A flag indicating whether it is in train phrase or not. Default True.
        mode(str): ['upscale_in_train'(default) | 'downscale_in_infer'].

                           1. upscale_in_train(default), upscale the output at training time

                              - train: out = input * mask / ( 1.0 - dropout_prob )
                              - inference: out = input

                           2. downscale_in_infer, downscale the output at inference

                              - train: out = input * mask
                              - inference: out = input * (1.0 - dropout_prob)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout, has same shape and data type as `x` .


    Examples:
        We use ``p=0.5`` in the following description for simplicity.

        1. When ``axis=None`` , this is commonly used dropout, which dropout each element of x randomly.

        ..  code-block:: text

            Let's see a simple case when x is a 2d tensor with shape 2*3:
            [[1 2 3]
             [4 5 6]]
            we generate mask with the same shape as x, which is 2*3. The value of mask is
            sampled from a Bernoulli distribution randomly. For example, we may get such mask:
            [[0 1 0]
             [1 0 1]]
            So the output is obtained from elementwise multiply of x and mask:
            [[0 2 0]
             [4 0 6]]
            Using default setting, i.e. ``mode='upscale_in_train'`` ,
            if in training phase, the final upscale output is:
            [[0 4 0 ]
             [8 0 12]]
            if in test phase, the output is the same as input:
            [[1 2 3]
             [4 5 6]]
            we can also set ``mode='downscale_in_infer'`` , then
            if in training phase, the final output is:
            [[0 2 0]
             [4 0 6]]
            if in test phase, the scale output is:
            [[0.5 1.  1.5]
             [2.  2.5 3. ]]



        2. When ``axis!=None`` , this is useful for dropping whole channels from an image or sequence.

        ..  code-block:: text

            Let's see the simple case when x is a 2d tensor with shape 2*3 again:
            [[1 2 3]
             [4 5 6]]
            (1) If ``axis=0`` , this means the dropout is only performed in axis `0` .
                we generate mask with the shape 2*1. Only in axis `0` the value is randomly selected.
                For example, we may get such mask:
                [[1]
                 [0]]
                The output is obtained from elementwise multiply of x and mask. Doing that the mask will be
                broadcast from 2*1 to 2*3:
                [[1 1 1]
                 [0 0 0]]
                and the result after elementwise multiply is:
                [[1 2 3]
                 [0 0 0]]
                then we can do upscale or downscale according to the setting of other arguments.
            (2) If ``axis=1`` , this means the dropout is only performed in axis `1` .
                we generate mask with the shape 1*3. Only in axis `1` the value is randomly selected.
                For example, we may get such mask:
                [[1 0 1]]
                Doing elementwise multiply the mask will be broadcast from 1*3 to 2*3:
                [[1 0 1]
                 [1 0 1]]
                and the result after elementwise multiply is:
                [[1 0 3]
                 [4 0 6]]
            (3) What about ``axis=[0, 1]`` ? This means the dropout is performed in all axes of x,
                which is the same case as default setting ``axis=None`` .
            (4) You may note that logically `axis=None` means the dropout is performed in none axis of x,
                We generate mask with the shape 1*1. Whole input is randomly selected or dropped.
                For example, we may get such mask:
                [[0]]
                Doing elementwise multiply the mask will be broadcast from 1*1 to 2*3:
                [[0 0 0]
                 [0 0 0]]
                and the result after elementwise multiply is:
                [[0 0 0]
                 [0 0 0]]
                Actually this is not what we want because all elements may set to zero~

        When x is a 4d tensor with shape `NCHW`, we can set ``axis=[0,1]`` and the dropout will be performed in channel `N` and `C`, `H` and `W` is tied, i.e. paddle.nn.dropout(x, p, axis=[0,1]) . Please refer to ``paddle.nn.functional.dropout2d`` for more details.
        Similarly, when x is a 5d tensor with shape `NCDHW`, we can set ``axis=[0,1]`` to perform dropout3d. Please refer to ``paddle.nn.functional.dropout3d`` for more details.

        .. code-block:: python

            import paddle
            import numpy as np

            x = np.array([[1,2,3], [4,5,6]]).astype('float32')
            x = paddle.to_tensor(x)
            y_train = paddle.nn.functional.dropout(x, 0.5)
            y_test = paddle.nn.functional.dropout(x, 0.5, training=False) 
            y_0 = paddle.nn.functional.dropout(x, axis=0)
            y_1 = paddle.nn.functional.dropout(x, axis=1)
            y_01 = paddle.nn.functional.dropout(x, axis=[0,1])
            print(x)
            print(y_train)
            print(y_test)
            print(y_0)
            print(y_1)
            print(y_01)

    """
    # fast return for p == 0
    if p == 0:
        return x

    if not isinstance(p, (float, int)):
        raise TypeError("p argument should be a number")
    if p < 0 or p > 1:
        raise ValueError("p argument should between 0 and 1")
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")
    if axis and not isinstance(axis, (int, list, tuple)):
        raise TypeError("datatype of axis argument should be int or list")

    if axis == None:  # commonly used dropout
        seed = None
        mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

        if in_dygraph_mode():
            if default_main_program().random_seed != 0:
                seed = default_main_program().random_seed
            out, mask = _C_ops.dropout(
                x, 'dropout_prob', p, 'is_test', not training, 'fix_seed',
                seed is not None, 'seed', seed
                if seed is not None else 0, 'dropout_implementation', mode)
            return out

        helper = LayerHelper('dropout', **locals())
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'dropout')

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        mask = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)

        def get_attrs(prog, dropout_prob, is_test, seed):
            if (seed is None or seed == 0) and prog.random_seed != 0:
                seed = prog.random_seed
            attrs = {
                'dropout_prob': dropout_prob,
                'is_test': is_test,
                'fix_seed': seed is not None,
                'seed': seed if seed is not None else 0,
                'dropout_implementation': mode,
            }
            return attrs

        attrs = get_attrs(helper.main_program, p, not training, seed)

        helper.append_op(
            type='dropout',
            inputs={'X': [x]},
            outputs={'Out': [out],
                     'Mask': [mask]},
            attrs=attrs)
        return out
    else:  #sometimes called dropout_nd #TODO: optimize with c++
        if not in_dygraph_mode():
            check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'dropout')
        dtype = x.dtype
        keep_prob = 1 - p
        if training:
            if p == 1.:
                return paddle.scale(x, scale=0.)

            scale_input = paddle.scale(
                x, scale=1 / keep_prob) if mode == 'upscale_in_train' else x

            #get mask shape
            input_shape = x.shape
            if not in_dygraph_mode():
                input_shape_tensor = paddle.shape(x)
            drop_axes = [axis] if isinstance(axis, int) else list(axis)
            if min(drop_axes) < 0 or max(drop_axes) > len(input_shape) - 1:
                raise ValueError("axis value should be greater than or equal to 0 and less than dimensions of x:{}, but get axis value:{} " \
                                 .format(len(input_shape), max(drop_axes)))
            if len(drop_axes) > len(input_shape):
                raise ValueError(
                    "length of axis should not be greater than dimensions of x:{}, but get length of axis: {}".
                    format(len(input_shape), len(drop_axes)))
            mask_shape = [1] * len(input_shape)
            if not in_dygraph_mode():
                for i in drop_axes:
                    mask_shape[i] = input_shape_tensor[i]
            else:
                for i in drop_axes:
                    mask_shape[i] = input_shape[i]

            #get mask
            random_tensor = paddle.uniform(
                mask_shape, dtype='float32', min=0., max=1.0)
            p = layers.fill_constant(shape=[1], dtype='float32', value=p)
            keep_mask = paddle.greater_equal(random_tensor, p)

            scale_input = paddle.cast(scale_input, dtype)
            keep_mask = paddle.cast(keep_mask, dtype)
            ret = paddle.multiply(scale_input, keep_mask, name=name)
            return ret
        else:  # test
            ret = paddle.scale(
                x, scale=keep_prob) if mode == 'downscale_in_infer' else x
            return ret


def dropout2d(x, p=0.5, training=True, data_format='NCHW', name=None):
    """
    Randomly zero out entire channels (in the batched input 4d tensor with the shape `NCHW` ,
    a channel is a 2D feature map with the shape `HW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.

    See ``paddle.nn.functional.dropout`` for more details.

    Args:
        x (Tensor):  The input is 4-D Tensor with shape [N, C, H, W] or [N, H, W, C].
                     The data type is float32 or float64.
        p (float): Probability of setting units to zero. Default 0.5.
        training (bool): A flag indicating whether it is in train phrase or not. Default True.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from `NCHW` or `NHWC` . The default is `NCHW` . When it is `NCHW` , the data is stored in the order of: [batch_size, input_channels, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout2d, has same shape and data type as `x` .


    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = np.random.random(size=(2, 3, 4, 5)).astype('float32')
            x = paddle.to_tensor(x)
            y_train = paddle.nn.functional.dropout2d(x)  #train
            y_test = paddle.nn.functional.dropout2d(x, training=False) #test
            for i in range(2):
                for j in range(3):
                    print(x.numpy()[i,j,:,:])
                    print(y_train.numpy()[i,j,:,:]) # may all 0
                    print(y_test.numpy()[i,j,:,:])
    """
    input_shape = x.shape
    if len(input_shape) != 4:
        raise ValueError("dimensions of x should be 4, but received {} != 4"\
        .format(len(input_shape)))

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    return dropout(
        x,
        p=p,
        axis=[0, 1] if data_format == 'NCHW' else [0, 3],
        training=training,
        mode="upscale_in_train",
        name=name)


def dropout3d(x, p=0.5, training=True, data_format='NCDHW', name=None):
    """
    Randomly zero out entire channels (in the batched input 5d tensor with the shape `NCDHW` ,
    a channel is a 3D feature map with the shape `DHW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.

    See ``paddle.nn.functional.dropout`` for more details.

    Args:
        x (Tensor):  The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C].
                     The data type is float32 or float64.
        p (float): Probability of setting units to zero. Default 0.5.
        training (bool): A flag indicating whether it is in train phrase or not. Default True.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from ``NCDHW`` or ``NDHWC``. The default is ``NCDHW`` . When it is ``NCDHW`` , the data is stored in the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout3d, has same shape and data type with `x` .


    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = np.random.random(size=(2, 3, 4, 5, 6)).astype('float32')
            x = paddle.to_tensor(x)
            y_train = paddle.nn.functional.dropout3d(x)  #train
            y_test = paddle.nn.functional.dropout3d(x, training=False) #test
            print(x.numpy()[0,0,:,:,:])
            print(y_train.numpy()[0,0,:,:,:]) # may all 0
            print(y_test.numpy()[0,0,:,:,:])
    """

    input_shape = x.shape
    if len(input_shape) != 5:
        raise ValueError("dimensions of x should be 5, but received {} != 5" \
        .format(len(input_shape)))

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    return dropout(
        x,
        p=p,
        axis=[0, 1] if data_format == 'NCDHW' else [0, 4],
        training=training,
        mode="upscale_in_train",
        name=name)


def alpha_dropout(x, p=0.5, training=True, name=None):
    """
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property.
    For an input with zero mean and unit standard deviation, the output of Alpha Dropout
    maintains the original mean and standard deviation of the input.
    Alpha Dropout fits well to SELU activate function by randomly setting activations to the negative saturation value.

    Args:
        x (Tensor): The input tensor. The data type is float32 or float64.
        p (float | int): Probability of setting units to zero. Default 0.5.
        training (bool): A flag indicating whether it is in train phrase or not. Default True.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout, has same shape and data type as `x`.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = np.array([[-1, 1], [-1, 1]]).astype('float32')
            x = paddle.to_tensor(x)
            y_train = paddle.nn.functional.alpha_dropout(x, 0.5)
            y_test = paddle.nn.functional.alpha_dropout(x, 0.5, training=False)
            print(x)
            print(y_train)
            # [[-0.10721093, 1.6655989 ], [-0.7791938, -0.7791938]] (randomly)
            print(y_test)
    """
    if not isinstance(p, (float, int)):
        raise TypeError("p argument should be a float or int")
    if p < 0 or p > 1:
        raise ValueError("p argument should between 0 and 1")

    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'alpha_dropout')

    if training:
        if p == 1:
            return paddle.scale(x, scale=0.)
        #get transformation params
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        a = ((1 - p) * (1 + p * alpha_p**2))**-0.5
        b = -a * alpha_p * p

        dtype = x.dtype
        input_shape = x.shape

        #get mask
        random_tensor = paddle.uniform(
            input_shape, dtype='float32', min=0., max=1.0)
        p = layers.fill_constant(shape=[1], dtype='float32', value=p)
        keep_mask = paddle.greater_equal(random_tensor, p)
        keep_mask = paddle.cast(keep_mask, dtype)
        drop_mask = paddle.subtract(
            layers.fill_constant(
                shape=input_shape, dtype=dtype, value=1.),
            keep_mask)

        #apply mask
        b = layers.fill_constant(shape=[1], dtype=dtype, value=b)
        y = paddle.add(paddle.multiply(x, keep_mask),
                       paddle.scale(
                           drop_mask, scale=alpha_p))
        res = paddle.add(paddle.scale(y, scale=a), b, name=name)
        return res
    else:  # test
        return x


def pad(x, pad, mode='constant', value=0, data_format="NCHW", name=None):
    """
    Pad tensor according to 'pad' and 'mode'.
    If mode is 'constant' and length of pad is twice as length of x dimension,
    then the padding will be started from the first dimension and moved back onto x
    according to 'pad' and 'value'.
    If mode is 'reflect', pad[0] and pad[1] must be no greater
    than width-1. The height and depth dimension has the same condition.

    Parameters:
        x (Tensor): The input tensor with data type float32/double/int32/int64_t.
        pad (Tensor | List[int] | Tuple[int]): The padding size with data type int.
            If mode is 'constant' and length of pad is twice as length of x dimension, then x will 
            be padded from the first  dimension to the last dimension.
            Else: 1. If input dimension is 3, then the pad has the form (pad_left,
            pad_right). 2. If the input dimension is 4, then the pad has the form (pad_left, pad_right, 
            pad_top, pad_bottom). 3. If the input dimension is 5, then the pad has the form 
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        mode (str): Four modes: 'constant' (default), 'reflect', 'replicate', 'circular'.
            When in 'constant' mode, this op uses a constant value to pad the input tensor.
            When in 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
            When in 'replicate' mode, uses input boundaries to pad the input tensor.
            When in 'circular' mode, uses circular input to pad the input tensor.
            Default is 'constant'
        value (float32): The value to fill the padded areas in 'constant' mode . Default is 0.0
        data_format (str): An string from: "NCL", "NLC", NHWC", "NCHW", "NCDHW", "NDHWC". Specify the data format of
           the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
                    
    Returns: a Tensor padded according to pad and mode and data type is same as input.
    Return Type: Tensor

    Examples:
        .. code-block:: text

            x = [[[[[1., 2., 3.],
                    [4., 5., 6.]]]]]

            Case 0:
                pad = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                mode = 'constant'
                value = 0
                Out = [[[[[0., 0., 0.],
                          [1., 2., 3.],
                          [4., 5., 6.],
                          [0., 0., 0.]]]]]

            Case 1:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'constant'
                value = 0
                Out = [[[[[0. 0. 0. 0. 0. 0. 0.]
                          [0. 0. 1. 2. 3. 0. 0.]
                          [0. 0. 4. 5. 6. 0. 0.]
                          [0. 0. 0. 0. 0. 0. 0.]]]]]

            Case 2:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'reflect'
                Out = [[[[[6. 5. 4. 5. 6. 5. 4.]
                          [3. 2. 1. 2. 3. 2. 1.]
                          [6. 5. 4. 5. 6. 5. 4.]
                          [3. 2. 1. 2. 3. 2. 1.]]]]]

            Case 3:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'replicate'
                Out = [[[[[1. 1. 1. 2. 3. 3. 3.]
                          [1. 1. 1. 2. 3. 3. 3.]
                          [4. 4. 4. 5. 6. 6. 6.]
                          [4. 4. 4. 5. 6. 6. 6.]]]]]

            Case 4:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'circular'
                Out = [[[[[5. 6. 4. 5. 6. 4. 5.]
                          [2. 3. 1. 2. 3. 1. 2.]
                          [5. 6. 4. 5. 6. 4. 5.]
                          [2. 3. 1. 2. 3. 1. 2.]]]]]

    Code Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F
            
            # example 1
            x_shape = (1, 1, 3)
            x = paddle.arange(np.prod(x_shape), dtype="float32").reshape(x_shape) + 1
            y = F.pad(x, [0, 0, 0, 0, 2, 3], value=1, mode='constant', data_format="NCL")
            print(y)
            # [[[1. 1. 1. 2. 3. 1. 1. 1.]]]
            
            # example 2
            x_shape = (1, 1, 3)
            x = paddle.arange(np.prod(x_shape), dtype="float32").reshape(x_shape) + 1
            y = F.pad(x, [2, 3], value=1, mode='constant', data_format="NCL")
            print(y)
            # [[[1. 1. 1. 2. 3. 1. 1. 1.]]]
            
            # example 3
            x_shape = (1, 1, 2, 3)
            x = paddle.arange(np.prod(x_shape), dtype="float32").reshape(x_shape) + 1
            y = F.pad(x, [1, 2, 1, 1], value=1, mode='circular')
            print(y)
            # [[[[6. 4. 5. 6. 4. 5.]
            #    [3. 1. 2. 3. 1. 2.]
            #    [6. 4. 5. 6. 4. 5.]
            #    [3. 1. 2. 3. 1. 2.]]]]
    """
    assert mode in ['reflect', 'replicate', 'constant', 'circular'], \
            "mode should be one of constant, reflect, replicate, circular, but got {}.".format(mode)

    data_format = data_format.upper()
    assert data_format in ["NCL", "NCHW", "NCDHW", "NLC", "NHWC", "NDHWC"], \
        "data_format should be in one of [NCL, NCHW, NCDHW, NLC, NHWC, NDHWC], " \
        "but got {}".format(data_format)

    x_dim = len(x.shape)

    if mode == "constant" and isinstance(pad, (
            list, tuple)) and len(pad) == x_dim * 2:
        return layers.pad(x, pad, pad_value=value)

    assert x_dim in [
        3, 4, 5
    ], "input tesor dimension must be in [3, 4, 5] but got {}".format(x_dim)

    supported_format_map = {
        3: ["NCL", "NLC"],
        4: ["NCHW", "NHWC"],
        5: ["NCDHW", "NDHWC"],
    }
    assert data_format in supported_format_map[x_dim], \
    "input tensor dimension is {}, it's data format should be in {} but got {}".format(
        x_dim, supported_format_map[x_dim], data_format)

    unsqueezed_dim = []

    if isinstance(pad, Variable):
        if data_format in ["NCL", "NCHW", "NCDHW"]:
            data_format = "NCDHW"
            if x_dim == 3:
                pad = concat([zeros((4, ), dtype="int32"), pad], axis=0)
                unsqueezed_dim = [3, 4]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = concat([pad, zeros((2, ), dtype="int32")], axis=0)
                unsqueezed_dim = [2]
                x = unsqueeze(x, axis=unsqueezed_dim)
        elif data_format in ["NLC", "NHWC", "NDHWC"]:
            data_format = "NDHWC"
            if x_dim == 3:
                pad = concat([zeros((4, ), dtype="int32"), pad], axis=0)
                unsqueezed_dim = [2, 3]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = concat([pad, zeros((2, ), dtype="int32")], axis=0)
                unsqueezed_dim = [1]
                x = unsqueeze(x, axis=unsqueezed_dim)
    else:
        pad = list(pad)
        if data_format in ["NCL", "NCHW", "NCDHW"]:
            data_format = "NCDHW"
            if x_dim == 3:
                pad = [0, 0, 0, 0] + pad
                unsqueezed_dim = [3, 4]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = pad + [0, 0]
                unsqueezed_dim = [2]
                x = unsqueeze(x, axis=unsqueezed_dim)
        elif data_format in ["NLC", "NHWC", "NDHWC"]:
            data_format = "NDHWC"
            if x_dim == 3:
                pad = [0, 0, 0, 0] + pad
                unsqueezed_dim = [2, 3]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = pad + [0, 0]
                unsqueezed_dim = [1]
                x = unsqueeze(x, axis=unsqueezed_dim)

    if in_dygraph_mode():
        if isinstance(pad, Variable):
            pad = pad.numpy()
        out = _C_ops.pad3d(x, "paddings", pad, "mode", mode, "value", value,
                           "data_format", data_format, "name", name)
    else:
        attrs = {'mode': mode, 'value': value, 'data_format': data_format}
        inputs = {'X': [x]}
        if isinstance(pad, Variable):
            inputs['Paddings'] = [pad]
            attrs['paddings'] = []
        else:
            attrs['paddings'] = pad

        helper = LayerHelper('pad3d', **locals())

        dtype = helper.input_dtype(input_param_name='input')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='pad3d', inputs=inputs, outputs={"Out": out}, attrs=attrs)

    if len(unsqueezed_dim) != 0:
        out = squeeze(out, axis=unsqueezed_dim)

    return out


def cosine_similarity(x1, x2, axis=1, eps=1e-8):
    """
    Compute cosine similarity between x1 and x2 along axis.

    Parameters:
        x1 (Tensor): First input. float32/double.
        x2 (Tensor): Second input. float32/double.
        axis (int): Dimension of vectors to compute cosine similarity. Default is 1.
        eps(float): Small value to avoid division by zero. Default is 1e-8.
                    
    Returns: a Tensor representing cosine similarity between x1 and x2 along axis.
    Return Type: Tensor

    Examples:
        .. code-block:: text

            Case 0:
                x1 = [[0.8024077  0.9927354  0.27238318 0.8344984 ]
                     [0.48949873 0.5797396  0.65444374 0.66510963]
                     [0.1031398  0.9614342  0.08365563 0.6796464 ]
                     [0.10760343 0.7461209  0.7726148  0.5801006 ]]
                x2 = [[0.62913156 0.1536727  0.9847992  0.04591406]
                     [0.9098952  0.15715368 0.8671125  0.3156102 ]
                     [0.4427798  0.54136837 0.5276275  0.32394758]
                     [0.3769419  0.8535014  0.48041078 0.9256797 ]]
                axis = 1
                eps = 1e-8
                Out: [0.5275037  0.8368967  0.75037485 0.9245899]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np

            np.random.seed(0)
            x1 = np.random.rand(2,3)
            x2 = np.random.rand(2,3)
            x1 = paddle.to_tensor(x1)
            x2 = paddle.to_tensor(x2)
            result = paddle.nn.functional.cosine_similarity(x1, x2, axis=0)
            print(result)
            # [0.99806249 0.9817672  0.94987036]
            
    """
    w12 = sum(paddle.multiply(x1, x2), axis=axis)
    w1 = sum(paddle.multiply(x1, x1), axis=axis)
    w2 = sum(paddle.multiply(x2, x2), axis=axis)
    n12 = sqrt(clip(w1 * w2, min=eps * eps))
    cos_sim = w12 / n12
    return cos_sim


def linear(x, weight, bias=None, name=None):
    r"""

    Fully-connected linear transformation operator. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    If the weight is a 2-D tensor of shape :math:`[in\_features, out\_features]` ,
    input should be a multi-dimensional tensor of shape
    :math:`[batch\_size, *, in\_features]` , where :math:`*` means any number of
    additional dimensions. The linear operator multiplies input tensor with
    weight and produces an output tensor of shape :math:`[batch\_size, *, out\_features]` , 
    If :math:`bias` is not None, the bias should be a 1-D tensor of shape
    :math:`[out\_features]` and will be added to the output.

    Parameters:
        x (Tensor): Input tensor. The data type should be float16, float32 or float64.
        weight (Tensor): Weight tensor. The data type should be float16, float32 or float64.
        bias (Tensor, optional): Bias tensor. The data type should be float16, float32 or float64.
                                 If it is set to None, no bias will be added to the output units.
        name (str, optional): Normally there is no need for user to set this parameter.
                              For detailed information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, the shape is :math:`[batch\_size, *, out\_features]` and the
        data type is the same with input :math:`x` .

    Examples:
        .. code-block:: python
          
          import paddle
          
          x = paddle.randn((3, 2), dtype="float32")
          # x: [[-0.32342386 -1.200079  ]
          #     [ 0.7979031  -0.90978354]
          #     [ 0.40597573  1.8095392 ]]
          weight = paddle.full(shape=[2, 4], fill_value="0.5", dtype="float32", name="weight")
          # weight: [[0.5 0.5 0.5 0.5]
          #          [0.5 0.5 0.5 0.5]]
          bias = paddle.ones(shape=[4], dtype="float32", name="bias")
          # bias: [1. 1. 1. 1.]
          y = paddle.nn.functional.linear(x, weight, bias)
          # y: [[0.23824859 0.23824859 0.23824859 0.23824859]
          #     [0.9440598  0.9440598  0.9440598  0.9440598 ]
          #     [2.1077576  2.1077576  2.1077576  2.1077576 ]]
    """
    if in_dygraph_mode():
        pre_bias = _C_ops.matmul_v2(x, weight, 'trans_x', False, 'trans_y',
                                    False)

        if bias is None:
            return pre_bias

        return _C_ops.elementwise_add(pre_bias, bias)
    else:
        helper = LayerHelper('linear', **locals())
        dtype = x.dtype

        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'linear')
        check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'], 'linear')

        inputs = {'X': [x], 'Y': [weight]}
        attrs = {'trans_x': False, 'trans_y': False}
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='matmul_v2', inputs=inputs, outputs={'Out': tmp}, attrs=attrs)
        if bias is not None:
            res = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_add',
                inputs={'X': [tmp],
                        'Y': [bias]},
                outputs={'Out': [res]},
                attrs={'axis': len(x.shape) - 1})
        else:
            res = tmp
        return res


def label_smooth(label, prior_dist=None, epsilon=0.1, name=None):
    r"""
    Label smoothing is a mechanism to regularize the classifier layer and is called
    label-smoothing regularization (LSR).

    Label smoothing is proposed to encourage the model to be less confident,
    since optimizing the log-likelihood of the correct label directly may
    cause overfitting and reduce the ability of the model to adapt. Label
    smoothing replaces the ground-truth label :math:`y` with the weighted sum
    of itself and some fixed distribution :math:`\mu`. For class :math:`k`,
    i.e.

    .. math::

        \\tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

    where :math:`1 - \epsilon` and :math:`\epsilon` are the weights
    respectively, and :math:`\\tilde{y}_k` is the smoothed label. Usually
    uniform distribution is used for :math:`\mu`.

    See more details about label smoothing in https://arxiv.org/abs/1512.00567.

    Parameters:
        label(Tensor): The input variable containing the label data. The
                        label data should use one-hot representation. It's
                        a multidimensional tensor with a shape of
                        :math:`[N_1, ..., Depth]`, where Depth is class number. The dtype can be "float32" and "float64".
        prior_dist(Tensor, optional): The prior distribution to be used to smooth
                        labels. If not provided, an uniform distribution
                        is used. It's a multidimensional tensor with a shape of
                        :math:`[1, class\_num]` . The default value is None.
        epsilon(float, optional): The weight used to mix up the original ground-truth
                        distribution and the fixed distribution. The default value is
                        0.1.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor containing the smoothed labels.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            
            x_data = np.array([[[0, 1, 0],
                                [ 1,  0, 1]]]).astype("float32")
            print(x_data.shape)
            paddle.disable_static()
            x = paddle.to_tensor(x_data, stop_gradient=False)
            output = paddle.nn.functional.label_smooth(x)
            print(output)
            
            #[[[0.03333334 0.93333334 0.03333334]
            #  [0.93333334 0.03333334 0.93333334]]]
    """
    if epsilon > 1. or epsilon < 0.:
        raise ValueError("The value of epsilon must be between 0 and 1.")

    if in_dygraph_mode():
        return _C_ops.label_smooth(label, prior_dist, 'epsilon', float(epsilon))

    check_variable_and_dtype(label, 'label', ['float32', 'float64'],
                             'label_smooth')

    helper = LayerHelper("label_smooth", **locals())
    label.stop_gradient = True
    smooth_label = helper.create_variable_for_type_inference(label.dtype)
    helper.append_op(
        type="label_smooth",
        inputs={"X": label,
                "PriorDist": prior_dist} if prior_dist else {"X": label},
        outputs={"Out": smooth_label},
        attrs={"epsilon": float(epsilon)})
    return smooth_label


def class_center_sample(label, num_classes, num_samples, group=None):
    """
    Class center sample method is proposed from the paper PartialFC that only sample a subset of the class centers.
    The process of sampling subset class centers is straightforward: 

    1. First select the positive class centers;
    2. Then randomly sample negative class centers.

    Specifically, given a label tensor, shape [batch_size], select all the positive class centers and randomly 
    sample negative class centers, then remap the input label tensor using the sampled class centers.

    For more information, Partial FC: Training 10 Million Identities on a Single Machine
    arxiv: https://arxiv.org/abs/2010.05222
    
    .. hint::
        If the number of the positive class centers is greater than the input num_samples, it keeps all the positive 
        class centers and the shape of sampled_class_center will be [num_positive_class_centers].
    
        The API supports CPU, single GPU and multi GPU.

    Args:
        label (Tensor): 1-D tensor with shape [N], each label in [0, num_classes)
        num_classes (int): A positive integer to specify the number of classes at local rank.
            Note that num_classes of each GPU can be different.
        num_samples (int): A positive integer to specify the number of class center to sample.
        group (Group, optional): The abstract representation of group.
            See paddle.distributed.collective.Group. Default is ``None``.

    Returns:
        Tuple of two ``Tensor`` : (remapped_label, sampled_class_center), remapped label using sampled class center,
        sampled class center from [0, num_classes).

    Examples:

    .. code-block:: python
        :name: code-example1

        # CPU or single GPU
        import paddle
        num_classes = 20
        batch_size = 10
        num_samples = 6
        label = paddle.randint(low=0, high=num_classes, shape=[batch_size], dtype='int64')
        remapped_label, sampled_class_index = paddle.nn.functional.class_center_sample(label, num_classes, num_samples)

        print(label)
        print(remapped_label)
        print(sampled_class_index)

        # the output is
        #Tensor(shape=[10], dtype=int64, place=CPUPlace, stop_gradient=True,
        #       [11, 5 , 1 , 3 , 12, 2 , 15, 19, 18, 19])
        #Tensor(shape=[10], dtype=int64, place=CPUPlace, stop_gradient=True,
        #       [4, 3, 0, 2, 5, 1, 6, 8, 7, 8])
        #Tensor(shape=[9], dtype=int64, place=CPUPlace, stop_gradient=True,
        #       [1 , 2 , 3 , 5 , 11, 12, 15, 18, 19])

    .. code-block:: python
        :name: code-example2

        # required: distributed
        # Multi GPU, test_class_center_sample.py
        import paddle
        import paddle.distributed as dist
        strategy = dist.fleet.DistributedStrategy()
        dist.fleet.init(is_collective=True, strategy=strategy)
        batch_size = 10
        num_samples = 6
        rank_id = dist.get_rank()
        # num_classes of each GPU can be different, e.g num_classes_list = [10, 8]
        num_classes_list = [10, 10]
        num_classes = paddle.sum(paddle.to_tensor(num_classes_list))
        label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64')
        label_list = []
        dist.all_gather(label_list, label)
        label = paddle.concat(label_list, axis=0)
        remapped_label, sampled_class_index = paddle.nn.functional.class_center_sample(label, num_classes_list[rank_id], num_samples)

        print(label)
        print(remapped_label)
        print(sampled_class_index)

        #python -m paddle.distributed.launch --gpus=0,1 test_class_center_sample.py
        # rank 0 output:
        #Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        #       [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
        #Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        #       [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
        #Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        #       [0, 2, 4, 8, 9, 3])
        
        # rank 1 output:
        #Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
        #       [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
        #Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
        #       [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
        #Tensor(shape=[7], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
        #       [0, 1, 2, 3, 5, 7, 8])
    """
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id
    rank = 0
    nranks = 1
    if core.is_compiled_with_dist():
        parallel_env = paddle.distributed.ParallelEnv()
        global_rank = parallel_env.rank
        rank = global_rank if group is None else group.get_group_rank(
            global_rank)
        nranks = parallel_env.world_size if group is None else group.nranks

    if num_samples > num_classes:
        raise ValueError(
            'Expected num_samples less than or equal to {}, got num_samples {}'.
            format(num_classes, num_samples))

    label_size = 1
    for dim in list(label.shape):
        label_size *= dim
    if label_size != -1 and label_size < 1:
        raise ValueError('Expected label_size > 0 \
             (got label_size: {})'.format(label_size))

    label_dims = len(list(label.shape))
    if label_dims != 1:
        raise ValueError('Expected label_dims == 1 \
             (got label_dims: {})'.format(label_dims))

    seed = None
    if (seed is None or seed == 0) and default_main_program().random_seed != 0:
        seed = default_main_program().random_seed

    if in_dygraph_mode():
        remapped_label, sampled_class_center = core.ops.class_center_sample(
            label, 'num_classes', num_classes, 'num_samples', num_samples,
            'ring_id', ring_id, 'nranks', nranks, 'rank', rank, 'fix_seed',
            seed is not None, 'seed', seed if seed is not None else 0)
        return remapped_label, sampled_class_center

    check_variable_and_dtype(label, 'label', ['int64', 'int32'],
                             'class_center_sample')
    op_type = 'class_center_sample'
    helper = LayerHelper(op_type, **locals())
    remapped_label = helper.create_variable_for_type_inference(
        dtype=label.dtype)
    sampled_class_center = helper.create_variable_for_type_inference(
        dtype=label.dtype)
    helper.append_op(
        type=op_type,
        inputs={'Label': label},
        outputs={
            'RemappedLabel': remapped_label,
            'SampledLocalClassCenter': sampled_class_center
        },
        attrs={
            'num_classes': num_classes,
            'num_samples': num_samples,
            'ring_id': ring_id,
            'nranks': nranks,
            'rank': rank,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0
        })
    return remapped_label, sampled_class_center
