#include "paddle/extension.h"
#include <vector>

using namespace std;
using namespace paddle;

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

class WarpMLS {
public:
  WarpMLS(const Tensor &src, const Tensor &src_pts, const Tensor &dst_pts,
          int64_t dst_w, int64_t dst_h, float trans_ratio = 1.0)
      : src(src), src_pts(src_pts), dst_pts(dst_pts), dst_w(dst_w),
        dst_h(dst_h), trans_ratio(trans_ratio) {
    CHECK_INPUT(src);
    CHECK_INPUT(src_pts);
    CHECK_INPUT(src_pts);
    rdx = zeros({dst_h, dst_w});
    rdy = zeros({dst_h, dst_w});
    pt_count = dst_pts.shape()[0];
  }

  inline Tensor BilinearInterp(Tensor x, Tensor y, float v11, float v12,
                               float v21, float v22) {
    return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x;
  }

  inline Tensor BilinearInterp(Tensor x, Tensor y, Tensor v11, Tensor v12,
                               Tensor v21, Tensor v22) {
    return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x;
  }

  void CalcDelta() {
    Tensor w = zeros({pt_count}, DataType::FLOAT32);
    auto w_data = w.data<float>();

    if (pt_count < 2)
      return;
    auto *src_pts_data = src_pts.data<int64_t>();
    auto *dst_pts_data = dst_pts.data<float>();

    int i = 0;
    while (true) {
      if (dst_w <= i && i < dst_w + grid_size - 1) {
        i = dst_w - 1;
      } else if (i >= dst_w) {
        break;
      }
      int j = 0;
      while (true) {
        if (dst_h <= j && j < dst_h + grid_size - 1) {
          j = dst_h - 1;
        } else if (j >= dst_h) {
          break;
        }

        float sw = 0;
        Tensor swp = zeros({2}, DataType::FLOAT32);
        Tensor swq = zeros({2}, DataType::FLOAT32);
        Tensor new_pt = zeros({2}, DataType::FLOAT32);
        Tensor cur_pt = zeros({2}, DataType::FLOAT32);
        auto *new_pt_data = new_pt.data<float>();
        auto *cur_pt_data = cur_pt.data<float>();
        cur_pt_data[0] = static_cast<float>(i);
        cur_pt_data[1] = static_cast<float>(j);

        int k = 0;
        for (; k < pt_count; ++k) {
          if (i == dst_pts_data[2 * k] && j == dst_pts_data[2 * k + 1])
            break;

          w_data[k] =
              1.0 /
              ((i - dst_pts_data[2 * k]) * (i - dst_pts_data[2 * k]) +
               (j - dst_pts_data[2 * k + 1]) * (j - dst_pts_data[2 * k + 1]));

          sw += w_data[k];
          Tensor tmp_dst = zeros({2}, DataType::FLOAT32);
          auto *tmp_dst_data = tmp_dst.data<float>();
          tmp_dst_data[0] = dst_pts_data[2 * k];
          tmp_dst_data[1] = dst_pts_data[2 * k + 1];

          Tensor tmp_src = zeros({2}, DataType::INT64);
          auto *tmp_src_data = tmp_src.data<int64_t>();
          tmp_src_data[0] = src_pts_data[2 * k];
          tmp_src_data[1] = src_pts_data[2 * k + 1];

          swp = swp + w_data[k] * tmp_dst;
          swq =
              swq + w_data[k] * experimental::cast(tmp_src, DataType::FLOAT32);
        }
        k--;

        if (k == pt_count - 1) {
          Tensor pstar = zeros({2}, DataType::FLOAT32);
          Tensor qstar = zeros({2}, DataType::FLOAT32);

          pstar = 1.0 / sw * swp;
          qstar = 1.0 / sw * swq;

          auto *pstar_data = pstar.data<float>();
          auto *qstar_data = qstar.data<float>();

          float miu_s = 0.0f;
          for (int k = 0; k < pt_count; ++k) {
            if (i == dst_pts_data[2 * k] && j == dst_pts_data[2 * k + 1])
              continue;
            float pt_i0 = dst_pts_data[2 * k] - pstar_data[0];
            float pt_i1 = dst_pts_data[2 * k + 1] - pstar_data[1];
            miu_s += w_data[k] * (pt_i0 * pt_i0 + pt_i1 * pt_i1);
          }

          cur_pt_data[0] = cur_pt_data[0] - pstar_data[0];
          cur_pt_data[1] = cur_pt_data[1] - pstar_data[1];

          float cur_pt_j0 = -cur_pt_data[1];
          float cur_pt_j1 = cur_pt_data[0];

          for (int k = 0; k < pt_count; ++k) {
            if (i == dst_pts_data[2 * k] && j == dst_pts_data[2 * k + 1])
              continue;

            float pt_i0 = dst_pts_data[2 * k] - pstar_data[0];
            float pt_i1 = dst_pts_data[2 * k + 1] - pstar_data[1];

            float pt_j0 = -pt_i1;
            float pt_j1 = pt_i0;

            Tensor tmp_pt = zeros({2}, DataType::FLOAT32);
            auto *tmp_pt_data = tmp_pt.data<float>();

            tmp_pt_data[0] = (pt_i0 * cur_pt_data[0] + pt_i1 * cur_pt_data[1]) *
                                 src_pts_data[2 * k] -
                             (pt_j0 * cur_pt_data[0] + pt_j1 * cur_pt_data[1]) *
                                 src_pts_data[2 * k + 1];

            tmp_pt_data[1] =
                (pt_j0 * cur_pt_j0 + pt_j1 * cur_pt_j1) *
                    src_pts_data[2 * k + 1] -
                (pt_i0 * cur_pt_j0 + pt_i1 * cur_pt_j1) * src_pts_data[2 * k];

            tmp_pt_data[0] = tmp_pt_data[0] * (w_data[k] / miu_s);
            tmp_pt_data[1] = tmp_pt_data[1] * (w_data[k] / miu_s);

            new_pt_data[0] = new_pt_data[0] + tmp_pt_data[0];
            new_pt_data[1] = new_pt_data[1] + tmp_pt_data[1];
          }
          new_pt_data[0] = new_pt_data[0] + qstar_data[0];
          new_pt_data[1] = new_pt_data[1] + qstar_data[1];
        } else {
          new_pt_data[0] = src_pts_data[2 * k];
          new_pt_data[1] = src_pts_data[2 * k + 1];
        }

        auto *rdx_data = rdx.data<float>();
        auto *rdy_data = rdy.data<float>();
        rdx_data[j * dst_w + i] = new_pt_data[0] - i;
        rdy_data[j * dst_w + i] = new_pt_data[1] - j;
        j += grid_size;
      }
      i += grid_size;
    }
  }

  std::vector<Tensor> GenImg() {

    int src_h = src.shape()[0];
    int src_w = src.shape()[1];

    Tensor dst = experimental::zeros_like(src, DataType::FLOAT32);
    auto *rdx_data = rdx.data<float>();
    auto *rdy_data = rdy.data<float>();
    auto *dst_data = dst.data<float>();
    auto *src_data = src.data<uint8_t>();

    for (int i = 0; i < dst_h; i += grid_size) {
      for (int j = 0; j < dst_w; j += grid_size) {
        int ni = i + grid_size;
        int nj = j + grid_size;
        int w = grid_size;
        int h = grid_size;
        if (ni >= dst_h) {
          ni = dst_h - 1;
          h = ni - i + 1;
        }
        if (nj >= dst_w) {
          nj = dst_w - 1;
          w = nj - j + 1;
        }

        Tensor h_tensor = full({1, 1}, h);
        Tensor w_tensor = full({1, 1}, w);
        Tensor start_tensor = full({1, 1}, 0);
        Tensor step_tensor = full({1, 1}, 1);
        Tensor di =
            reshape(experimental::arange(start_tensor, h_tensor, step_tensor,
                                         DataType::FLOAT32, CPUPlace()),
                    {-1, 1});
        Tensor dj =
            reshape(experimental::arange(start_tensor, w_tensor, step_tensor,
                                         DataType::FLOAT32, CPUPlace()),
                    {1, -1});

        Tensor delta_x = BilinearInterp(
            di / h, dj / w, rdx_data[i * dst_w + j], rdx_data[i * dst_w + nj],
            rdx_data[ni * dst_w + j], rdx_data[ni * dst_w + nj]);

        Tensor delta_y = BilinearInterp(
            di / h, dj / w, rdy_data[i * dst_w + j], rdy_data[i * dst_w + nj],
            rdy_data[ni * dst_w + j], rdy_data[ni * dst_w + nj]);
        Tensor nx = j + dj + delta_x * trans_ratio;
        Tensor ny = i + di + delta_y * trans_ratio;

        nx = clip(nx, 0, src_w - 1);
        ny = clip(ny, 0, src_h - 1);
        Tensor nxi = experimental::cast(floor(nx), DataType::INT32);
        Tensor nyi = experimental::cast(floor(ny), DataType::INT32);
        Tensor nxi1 = experimental::cast(ceil(nx), DataType::INT32);
        Tensor nyi1 = experimental::cast(ceil(ny), DataType::INT32);
        Tensor x, y;
        if (src.shape().size() == 3) {
          x = tile(
              unsqueeze(experimental::cast(ny, DataType::INT32) - nyi, {-1}),
              {1, 1, 3});
          y = tile(
              unsqueeze(experimental::cast(nx, DataType::INT32) - nxi, {-1}),
              {1, 1, 3});
        } else {
          x = ny - nyi;
          y = nx - nxi;
        }

        auto *nxi_data = nxi.data<int>();
        auto *nyi_data = nyi.data<int>();
        auto *nxi1_data = nxi1.data<int>();
        auto *nyi1_data = nyi1.data<int>();

        Tensor src1 = empty({nx.shape()[0], nx.shape()[1], src.shape()[2]},
                            DataType::UINT8);
        Tensor src2 = empty({nx.shape()[0], nx.shape()[1], src.shape()[2]},
                            DataType::UINT8);
        Tensor src3 = empty({nx.shape()[0], nx.shape()[1], src.shape()[2]},
                            DataType::UINT8);
        Tensor src4 = empty({nx.shape()[0], nx.shape()[1], src.shape()[2]},
                            DataType::UINT8);

        auto *src1_data = src1.data<uint8_t>();
        auto *src2_data = src2.data<uint8_t>();
        auto *src3_data = src3.data<uint8_t>();
        auto *src4_data = src4.data<uint8_t>();

        int dim0 = src1.shape()[0];
        int dim1 = src1.shape()[1];
        int dim2 = src1.shape()[2];

        int src_dim = src.shape()[1];

        for (int i = 0; i < dim0; i++) {
          for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
              src1_data[i * dim1 * dim2 + j * dim2 + k] =
                  src_data[nyi_data[i * dim1 + j] * src_dim * dim2 +
                           nxi_data[i * dim1 + j] * dim2 + k];

              src2_data[i * dim1 * dim2 + j * dim2 + k] =
                  src_data[nyi_data[i * dim1 + j] * src_dim * dim2 +
                           nxi1_data[i * dim1 + j] * dim2 + k];

              src3_data[i * dim1 * dim2 + j * dim2 + k] =
                  src_data[nyi1_data[i * dim1 + j] * src_dim * dim2 +
                           nxi_data[i * dim1 + j] * dim2 + k];

              src4_data[i * dim1 * dim2 + j * dim2 + k] =
                  src_data[nyi1_data[i * dim1 + j] * src_dim * dim2 +
                           nxi1_data[i * dim1 + j] * dim2 + k];
            }
          }
        }

        Tensor res =
            BilinearInterp(x, y, experimental::cast(src1, DataType::INT32),
                           experimental::cast(src2, DataType::INT32),
                           experimental::cast(src3, DataType::INT32),
                           experimental::cast(src4, DataType::INT32));

        int dst_dim0 = dst.shape()[0];
        int dst_dim1 = dst.shape()[1];
        int dst_dim2 = dst.shape()[2];
        auto *res_data = res.data<int32_t>();
        dim0 = res.shape()[0];
        dim1 = res.shape()[1];
        dim2 = res.shape()[2];

        for (int ii = i, iii = 0; ii < i + h; ii++, iii++) {
          for (int jj = j, jjj = 0; jj < j + w; jj++, jjj++) {
            for (int kk = 0; kk < dst_dim2; kk++) {
              dst_data[ii * dst_dim1 * dst_dim2 + jj * dst_dim2 + kk] =
                  res_data[iii * dim1 * dim2 + jjj * dim2 + kk];
            }
          }
        }
      }
    }
    dst = clip(experimental::cast(dst, DataType::INT32), 0, 255);
    dst = experimental::cast(dst, DataType::UINT8);
    return {dst};
  }

  std::vector<Tensor> Generate() {
    CalcDelta();
    return GenImg();
  }

private:
  Tensor src;
  Tensor src_pts;
  Tensor dst_pts;
  int64_t dst_w;
  int64_t dst_h;
  int grid_size{100};
  float trans_ratio{1.0};
  Tensor rdx;
  Tensor rdy;
  int64_t pt_count;
};

std::vector<Tensor> RunCPUWarpMLS(const Tensor &src, const Tensor &src_pts,
                                  const Tensor &dst_pts, int64_t dst_w,
                                  int64_t dst_h, float trans_ratio = 1.0) {
  WarpMLS warp_mls(src, src_pts, dst_pts, dst_w, dst_h, trans_ratio);
  return warp_mls.Generate();
}

PD_BUILD_OP(warp_mls)
    .Inputs({"X", "Y", "Z"})
    .Outputs({"Out"})
    .Attrs({"dst_w: int64_t", "dst_h: int64_t", "trans_ratio: float"})
    .SetKernelFn(PD_KERNEL(RunCPUWarpMLS));