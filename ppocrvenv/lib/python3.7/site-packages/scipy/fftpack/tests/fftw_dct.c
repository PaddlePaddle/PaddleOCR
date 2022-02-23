#include <stdlib.h>
#include <stdio.h>

#include <fftw3.h>

#if DCT_TEST_PRECISION == 1
typedef float float_prec;
#define PF "%.7f"
#define FFTW_PLAN fftwf_plan
#define FFTW_MALLOC fftwf_malloc
#define FFTW_FREE fftwf_free
#define FFTW_PLAN_CREATE fftwf_plan_r2r_1d
#define FFTW_EXECUTE fftwf_execute
#define FFTW_DESTROY_PLAN fftwf_destroy_plan
#define FFTW_CLEANUP fftwf_cleanup
#elif DCT_TEST_PRECISION == 2
typedef double float_prec;
#define PF "%.18f"
#define FFTW_PLAN fftw_plan
#define FFTW_MALLOC fftw_malloc
#define FFTW_FREE fftw_free
#define FFTW_PLAN_CREATE fftw_plan_r2r_1d
#define FFTW_EXECUTE fftw_execute
#define FFTW_DESTROY_PLAN fftw_destroy_plan
#define FFTW_CLEANUP fftw_cleanup
#elif DCT_TEST_PRECISION == 3
typedef long double float_prec;
#define PF "%.18Lf"
#define FFTW_PLAN fftwl_plan
#define FFTW_MALLOC fftwl_malloc
#define FFTW_FREE fftwl_free
#define FFTW_PLAN_CREATE fftwl_plan_r2r_1d
#define FFTW_EXECUTE fftwl_execute
#define FFTW_DESTROY_PLAN fftwl_destroy_plan
#define FFTW_CLEANUP fftwl_cleanup
#else
#error DCT_TEST_PRECISION must be a number 1-3
#endif


enum type {
        DCT_I = 1,
        DCT_II = 2,
        DCT_III = 3,
        DCT_IV = 4,
        DST_I = 5,
        DST_II = 6,
        DST_III = 7,
	    DST_IV = 8,
};

int gen(int type, int sz)
{
        float_prec *a, *b;
        FFTW_PLAN p;
        int i, tp;

        a = FFTW_MALLOC(sizeof(*a) * sz);
        if (a == NULL) {
                fprintf(stderr, "failure\n");
                exit(EXIT_FAILURE);
        }
        b = FFTW_MALLOC(sizeof(*b) * sz);
        if (b == NULL) {
                fprintf(stderr, "failure\n");
                exit(EXIT_FAILURE);
        }

        switch(type) {
                case DCT_I:
                        tp = FFTW_REDFT00;
                        break;
                case DCT_II:
                        tp = FFTW_REDFT10;
                        break;
                case DCT_III:
                        tp = FFTW_REDFT01;
                        break;
                case DCT_IV:
                        tp = FFTW_REDFT11;
                        break;
                case DST_I:
                        tp = FFTW_RODFT00;
                        break;
                case DST_II:
                        tp = FFTW_RODFT10;
                        break;
                case DST_III:
                        tp = FFTW_RODFT01;
                        break;
                case DST_IV:
                        tp = FFTW_RODFT11;
                        break;
                default:
                        fprintf(stderr, "unknown type\n");
                        exit(EXIT_FAILURE);
        }

        switch(type) {
            case DCT_I:
            case DCT_II:
            case DCT_III:
            case DCT_IV:
                for(i=0; i < sz; ++i) {
                    a[i] = i;
                }
                break;
            case DST_I:
            case DST_II:
            case DST_III:
            case DST_IV:
/*                TODO: what should we do for dst's?*/
                for(i=0; i < sz; ++i) {
                    a[i] = i;
                }
                break;
            default:
                fprintf(stderr, "unknown type\n");
                exit(EXIT_FAILURE);
        }

        p = FFTW_PLAN_CREATE(sz, a, b, tp, FFTW_ESTIMATE);
        FFTW_EXECUTE(p);
        FFTW_DESTROY_PLAN(p);

        for(i=0; i < sz; ++i) {
                printf(PF"\n", b[i]);
        }
        FFTW_FREE(b);
        FFTW_FREE(a);

        return 0;
}

int main(int argc, char* argv[])
{
        int n, tp;

        if (argc < 3) {
                fprintf(stderr, "missing argument: program type n\n");
                exit(EXIT_FAILURE);
        }
        tp = atoi(argv[1]);
        n = atoi(argv[2]);

        gen(tp, n);
        FFTW_CLEANUP();

        return 0;
}
