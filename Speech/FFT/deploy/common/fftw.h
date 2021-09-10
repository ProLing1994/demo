#ifndef _ASR_FFTW_H_
#define _ASR_FFTW_H_

#include <fftw3.h>

class rm_FFTW
{
public:
	rm_FFTW(int n_fft);
	~rm_FFTW();

public:
	bool fft_double();
    bool psd_double(float * in, float * out, bool log_bool=false);

	bool fft_float();
    bool psd_float(float * in, float * out, bool log_bool=false);
private:
	int m_n_fft;

    // double
    double* m_fft_in_double;
    fftw_complex* m_fft_in_complex_double;
    fftw_complex* m_fft_out_complex_double;
    fftw_plan m_fft_plan_double;

    // float
    float* m_fft_in_float;
    fftwf_complex* m_fft_in_complex_float;
    fftwf_complex* m_fft_out_complex_float;
    fftwf_plan m_fft_plan_float;
};

#endif // _ASR_FFTW_H_