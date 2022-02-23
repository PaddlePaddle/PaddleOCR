CC	= gcc
LD	= gcc

fftw_single: fftw_dct.c
	$(CC) -W -Wall -DDCT_TEST_USE_SINGLE $< -o $@ -lfftw3f

fftw_double: fftw_dct.c
	$(CC) -W -Wall $< -o $@ -lfftw3

clean:
	rm -f fftw_single
	rm -f fftw_double
	rm -f *.o
