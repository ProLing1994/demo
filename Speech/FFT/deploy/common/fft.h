#ifndef _ASR_FFT_H_
#define _ASR_FFT_H_

#include <stdio.h>

#define MAX_MATRIX_SIZE 4194304 // 2048 * 2048
#define PI 3.141592653
#define MAX_VECTOR_LENGTH 10000 //

typedef struct Complex
{
	float rl;
	float im;
} Complex;

class rm_FFT
{
public:
	rm_FFT(int n_fft);
	~rm_FFT(void);

public:
	bool fft();
	bool ifft();
	void weight_init();
	void clear();
	bool is_power_of_two(int num);
	int get_computation_layers(int num); // calculate the layers of computation needed for FFT

	int m_n_fft;
	Complex *m_in_sequence = NULL;
	Complex *m_out_sequence = NULL;

private:
	// weight array
	Complex *m_Weights =NULL;
	Complex *m_pVec = NULL;
	Complex *m_X = NULL;

	double *m_W_rl = NULL;
	double *m_W_im = NULL;
	double *m_X_rl = NULL;
	double *m_X_im = NULL;
	double *m_X2_rl = NULL;
	double *m_X2_im = NULL;
};

#endif // _ASR_FFT_H_