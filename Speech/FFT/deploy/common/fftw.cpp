#include <algorithm>
#include <math.h>

#include "fftw.h"

rm_FFTW::rm_FFTW(int n_fft)
{
	m_n_fft = n_fft;

    // double 
    m_fft_in_double = (double*)fftw_malloc(sizeof(double) * m_n_fft);
    m_fft_in_complex_double = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * m_n_fft);
    m_fft_out_complex_double = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * m_n_fft);
    // m_fft_plan_double = fftw_plan_dft_1d(m_n_fft, m_fft_in_complex_double, m_fft_out_complex_double, FFTW_FORWARD, FFTW_ESTIMATE);
    // m_fft_plan_double = fftw_plan_dft_1d(m_n_fft, m_fft_in_complex_double, m_fft_out_complex_double, FFTW_FORWARD, FFTW_MEASURE);
    // m_fft_plan_double = fftw_plan_dft_r2c_1d(m_n_fft, m_fft_in_double, m_fft_out_complex_double, FFTW_ESTIMATE);
    m_fft_plan_double = fftw_plan_dft_r2c_1d(m_n_fft, m_fft_in_double, m_fft_out_complex_double, FFTW_MEASURE);

    // float
    m_fft_in_float = (float*)fftwf_malloc(sizeof(float) * m_n_fft);
    m_fft_in_complex_float = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * m_n_fft);
    m_fft_out_complex_float = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * m_n_fft);
    // m_fft_plan_float = fftwf_plan_dft_1d(m_n_fft, m_fft_in_complex_float, m_fft_out_complex_float, FFTW_FORWARD, FFTW_ESTIMATE);
    // m_fft_plan_float = fftwf_plan_dft_1d(m_n_fft, m_fft_in_complex_float, m_fft_out_complex_float, FFTW_FORWARD, FFTW_MEASURE);
    // m_fft_plan_float = fftwf_plan_dft_r2c_1d(m_n_fft, m_fft_in_float, m_fft_out_complex_float, FFTW_ESTIMATE);
    m_fft_plan_float = fftwf_plan_dft_r2c_1d(m_n_fft, m_fft_in_float, m_fft_out_complex_float, FFTW_MEASURE);
}

rm_FFTW::~rm_FFTW()
{
    // double 
    if(m_fft_in_complex_double != nullptr)
        fftw_free(m_fft_in_complex_double);
    if(m_fft_out_complex_double != nullptr)
        fftw_free(m_fft_out_complex_double);
    fftw_destroy_plan(m_fft_plan_double);

    // float
    if(m_fft_in_complex_float != nullptr)
        fftwf_free(m_fft_in_complex_float);
    if(m_fft_out_complex_float != nullptr)
        fftwf_free(m_fft_out_complex_float);
    fftwf_destroy_plan(m_fft_plan_float);
}

// Fourier transform
bool rm_FFTW::fft_double()
{
    fftw_execute(m_fft_plan_double);
	return true;
}

// Fourier transform
bool rm_FFTW::fft_float()
{
    fftwf_execute(m_fft_plan_float);
	return true;
}

// Power Spectral Density
bool rm_FFTW::psd_double(float * in, float * out, bool log_bool)
{
    for (int i = 0; i < m_n_fft; i++)
    {
        m_fft_in_complex_double[i][0] = in[i];
        // m_fft_in_double[i] = in[i];
    }

    fft_double();

    for (int i = 0; i < m_n_fft / 2; i++)
    {
        if (log_bool)
        {
            out[i] = log((sqrt(pow(m_fft_out_complex_double[i][0], 2) + pow(m_fft_out_complex_double[i][1], 2)) / m_n_fft) + 1);
        }
        else 
        {
            out[i] = sqrt(pow(m_fft_out_complex_double[i][0], 2) + pow(m_fft_out_complex_double[i][1], 2)) / m_n_fft;
        }
    }
    return true;
}

// Power Spectral Density
bool rm_FFTW::psd_float(float * in, float * out, bool log_bool)
{
    for (int i = 0; i < m_n_fft; i++)
    {
        // m_fft_in_complex_float[i][0] = in[i];
        m_fft_in_float[i] = in[i];
    }

    fft_float();

    for (int i = 0; i < m_n_fft / 2; i++)
    {
        if (log_bool)
        {
            out[i] = log((sqrt(pow(m_fft_out_complex_float[i][0], 2) + pow(m_fft_out_complex_float[i][1], 2)) / m_n_fft) + 1);
        }
        else 
        {
            out[i] = sqrt(pow(m_fft_out_complex_float[i][0], 2) + pow(m_fft_out_complex_float[i][1], 2)) / m_n_fft;
        }
    }
    return true;
}