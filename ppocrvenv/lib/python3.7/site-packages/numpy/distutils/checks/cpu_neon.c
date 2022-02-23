#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(void)
{
    float32x4_t v1 = vdupq_n_f32(1.0f), v2 = vdupq_n_f32(2.0f);
    int ret = (int)vgetq_lane_f32(vmulq_f32(v1, v2), 0);
#ifdef __aarch64__
    float64x2_t vd1 = vdupq_n_f64(1.0), vd2 = vdupq_n_f64(2.0);
    ret += (int)vgetq_lane_f64(vmulq_f64(vd1, vd2), 0);
#endif
    return ret;
}
