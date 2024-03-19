#include "bezier_align_op.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor>
BezieralignCPUForward(const paddle::Tensor &x, const paddle::Tensor &rois,
                      const paddle::Tensor &rois_num, const int pooled_height,
                      const int pooled_width, float spatial_scale,
                      float sampling_ratio, bool aligned) {
  //   CHECK_INPUT(x);

  auto out = paddle::empty({1, 3, 42, 320});

  PD_DISPATCH_FLOATING_TYPES(x.type(), "bezieralign_cpu_forward_kernel", ([&] {
                               bezieralign_cpu_forward_kernel<data_t>(
                                   x.data<data_t>(), rois.data<data_t>(),
                                   rois_num.data<data_t>(), out.data<data_t>(),
                                   pooled_height, pooled_width, spatial_scale,
                                   sampling_ratio, aligned);
                             }));

  return {out};
}

PD_BUILD_OP(custom_bezier)
    .Inputs({"X", "Rois", "Rois_num", "Pooledheight", "Pooledwidth",
             "Spatial_scale", "Sampling_ratio", "Aligned"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(BezieralignCPUForward));
