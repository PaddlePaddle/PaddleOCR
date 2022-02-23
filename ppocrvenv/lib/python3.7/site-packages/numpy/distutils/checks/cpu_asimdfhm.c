#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(void)
{
    float16x8_t vhp  = vdupq_n_f16((float16_t)1);
    float16x4_t vlhp = vdup_n_f16((float16_t)1);
    float32x4_t vf   = vdupq_n_f32(1.0f);
    float32x2_t vlf  = vdup_n_f32(1.0f);

    int ret  = (int)vget_lane_f32(vfmlal_low_u32(vlf, vlhp, vlhp), 0);
        ret += (int)vgetq_lane_f32(vfmlslq_high_u32(vf, vhp, vhp), 0);

    return ret;
}
