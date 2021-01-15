#ifndef _ASR_FFT_H_
#define _ASR_FFT_H_

#include <stdio.h>

#define MAX_MATRIX_SIZE 4194304 // 2048 * 2048
#define PI 3.141592653
#define MAX_VECTOR_LENGTH 10000 //
//#define      FFT_N                          512

typedef struct Complex
{
	float rl;
	float im;
} Complex;

class rm_FFT
{
public:
	rm_FFT(int FFT_N);
	~rm_FFT(void);

public:
	bool fft(const Complex inVec[], const int len, Complex outVec[]); // ���ڵ����㷨�Ŀ��ٸ���Ҷ�任
	bool ifft(const Complex ComplexinVec[], const int len, Complex outVec[]);
	bool is_power_of_two(int num);
	int get_computation_layers(int num); // calculate the layers of computation needed for FFT
	Complex *inVec = NULL;
	Complex *outVec = NULL;
};

#endif // _ASR_FFT_H_