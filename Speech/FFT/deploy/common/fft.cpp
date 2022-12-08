#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft.h"

rm_FFT::rm_FFT(int n_fft)
{
	m_n_fft = n_fft;
	m_in_sequence = new Complex[m_n_fft];	
	m_out_sequence = new Complex[m_n_fft];

	// weight array
	m_Weights = new Complex[m_n_fft];
	m_pVec = new Complex[m_n_fft];
	m_X = new Complex[m_n_fft];

	m_W_rl = new double[m_n_fft];
	m_W_im = new double[m_n_fft];
	m_X_rl = new double[m_n_fft];
	m_X_im = new double[m_n_fft];
	m_X2_rl = new double[m_n_fft];
	m_X2_im = new double[m_n_fft];

	// init
	memset(m_in_sequence, 0, sizeof(Complex) * m_n_fft);
	memset(m_out_sequence, 0, sizeof(Complex) * m_n_fft);

	memset(m_Weights, 0, sizeof(Complex) * m_n_fft);
	memset(m_pVec, 0, sizeof(Complex) * m_n_fft);
	memset(m_X, 0, sizeof(Complex) * m_n_fft);

	memset(m_W_rl, 0, sizeof(double) * m_n_fft);
	memset(m_W_im, 0, sizeof(double) * m_n_fft);
	memset(m_X_rl, 0, sizeof(double) * m_n_fft);
	memset(m_X_im, 0, sizeof(double) * m_n_fft);
	memset(m_X2_rl, 0, sizeof(double) * m_n_fft);
	memset(m_X2_im, 0, sizeof(double) * m_n_fft);

	// weight init
	weight_init();
}

rm_FFT::~rm_FFT()
{
	delete[] m_in_sequence;
	delete[] m_out_sequence;

	// weight array
	delete[] m_pVec;
	delete[] m_Weights;
	delete[] m_X;

	delete[] m_W_rl;
	delete[] m_W_im;
	delete[] m_X_rl;
	delete[] m_X_im;
	delete[] m_X2_rl;
	delete[] m_X2_im;
}

void rm_FFT::clear()
{
	memset(m_in_sequence, 0, sizeof(Complex) * m_n_fft);
	memset(m_out_sequence, 0, sizeof(Complex) * m_n_fft);
}

void rm_FFT::weight_init()
{
	double fixed_factor = (-2 * PI) / m_n_fft;
	for (int i = 0; i < m_n_fft / 2; i++)
	{
		double angle = i * fixed_factor;
		m_Weights[i].rl = cos(angle);
		m_Weights[i].im = sin(angle);
	}
	for (int i = m_n_fft / 2; i < m_n_fft; i++)
	{
		m_Weights[i].rl = -(m_Weights[i - m_n_fft / 2].rl);
		m_Weights[i].im = -(m_Weights[i - m_n_fft / 2].im);
	}
}

bool rm_FFT::is_power_of_two(int num)
{
	int temp = num;
	int mod = 0;
	int result = 0;

	if (num < 2)
		return false;
	if (num == 2)
		return true;

	while (temp > 1)
	{
		result = temp / 2;
		mod = temp % 2;
		if (mod)
			return false;
		if (2 == result)
			return true;
		temp = result;
	}
	return false;
}

int rm_FFT::get_computation_layers(int num)
{
	int nLayers = 0;
	int len = num;
	if (len == 2)
		return 1;
	while (true)
	{
		len = len / 2;
		nLayers++;
		if (len == 2)
			return nLayers + 1;
		if (len < 1)
			return -1;
	}
}

// Fourier transform
bool rm_FFT::fft()
{
	if ((m_n_fft <= 0) || (NULL == m_in_sequence) || (NULL == m_out_sequence))
		return false;
	if (!is_power_of_two(m_n_fft))
		return false;

	// init the array
	memset(m_pVec, 0, sizeof(Complex) * m_n_fft);
	memset(m_X, 0, sizeof(Complex) * m_n_fft);

	memcpy(m_pVec, m_in_sequence, m_n_fft * sizeof(Complex));

	int r = get_computation_layers(m_n_fft);

	// ���㵹��λ��
	int index = 0;
	for (int i = 0; i < m_n_fft; i++)
	{
		index = 0;
		for (int m = r - 1; m >= 0; m--)
		{
			index += (1 && (i & (1 << m))) << (r - m - 1);
		}
		m_X[i].rl = m_pVec[index].rl;
		m_X[i].im = m_pVec[index].im;
	}

	// ������ٸ���Ҷ�任
	for (int L = 1; L <= r; L++)
	{
		int distance = 1 << (L - 1);
		int W = 1 << (r - L);

		int B = m_n_fft >> L;
		int N = m_n_fft / B;

		for (int b = 0; b < B; b++)
		{
			int mid = b * N;
			for (int n = 0; n < N / 2; n++)
			{
				int index = n + mid;
				int dist = index + distance;
				m_pVec[index].rl = m_X[index].rl + (m_Weights[n * W].rl * m_X[dist].rl - m_Weights[n * W].im * m_X[dist].im); // Fe + W*Fo
				m_pVec[index].im = m_X[index].im + m_Weights[n * W].im * m_X[dist].rl + m_Weights[n * W].rl * m_X[dist].im;
			}
			for (int n = N / 2; n < N; n++)
			{
				int index = n + mid;
				m_pVec[index].rl = m_X[index - distance].rl + m_Weights[n * W].rl * m_X[index].rl - m_Weights[n * W].im * m_X[index].im; // Fe - W*Fo
				m_pVec[index].im = m_X[index - distance].im + m_Weights[n * W].im * m_X[index].rl + m_Weights[n * W].rl * m_X[index].im;
			}
		}

		memcpy(m_X, m_pVec, m_n_fft * sizeof(Complex));
	}

	memcpy(m_out_sequence, m_pVec, m_n_fft * sizeof(Complex));
	return true;
}

bool rm_FFT::ifft()
{
	if ((m_n_fft <= 0) || (!m_in_sequence))
		return false;
	if (false == is_power_of_two(m_n_fft))
	{
		return false;
	}

	// init the weight array
	memset(m_W_rl, 0, sizeof(double) * m_n_fft);
	memset(m_W_im, 0, sizeof(double) * m_n_fft);
	memset(m_X_rl, 0, sizeof(double) * m_n_fft);
	memset(m_X_im, 0, sizeof(double) * m_n_fft);
	memset(m_X2_rl, 0, sizeof(double) * m_n_fft);
	memset(m_X2_im, 0, sizeof(double) * m_n_fft);
	
	double fixed_factor = (-2 * PI) / m_n_fft;
	for (int i = 0; i < m_n_fft / 2; i++)
	{
		double angle = i * fixed_factor;
		m_W_rl[i] = cos(angle);
		m_W_im[i] = sin(angle);
	}
	for (int i = m_n_fft / 2; i < m_n_fft; i++)
	{
		m_W_rl[i] = -(m_W_rl[i - m_n_fft / 2]);
		m_W_im[i] = -(m_W_im[i - m_n_fft / 2]);
	}

	// ʱ������д��X1
	for (int i = 0; i < m_n_fft; i++)
	{
		m_X_rl[i] = m_in_sequence[i].rl;
		m_X_im[i] = m_in_sequence[i].im;
	}
	memset(m_X2_rl, 0, sizeof(double) * m_n_fft);
	memset(m_X2_im, 0, sizeof(double) * m_n_fft);

	int r = get_computation_layers(m_n_fft);
	if (-1 == r)
	{
		return false;
	}
	for (int L = r; L >= 1; L--)
	{
		int distance = 1 << (L - 1);
		int W = 1 << (r - L);

		int B = m_n_fft >> L;
		int N = m_n_fft / B;
		//sprintf(msg + 6, "B %d, N %d, W %d, distance %d, L %d", B, N, W, distance, L);
		//OutputDebugStringA(msg);

		for (int b = 0; b < B; b++)
		{
			for (int n = 0; n < N / 2; n++)
			{
				int index = n + b * N;
				m_X2_rl[index] = (m_X_rl[index] + m_X_rl[index + distance]) / 2;
				m_X2_im[index] = (m_X_im[index] + m_X_im[index + distance]) / 2;
				//sprintf(msg + 6, "%d, %d: %lf, %lf", n + 1, index, m_X2_rl[index], m_X2_im[index]);
				//OutputDebugStringA(msg);
			}
			for (int n = N / 2; n < N; n++)
			{
				int index = n + b * N;
				m_X2_rl[index] = (m_X_rl[index] - m_X_rl[index - distance]) / 2; // ��Ҫ�ٳ���W_rl[n*W]
				m_X2_im[index] = (m_X_im[index] - m_X_im[index - distance]) / 2;
				double square = m_W_rl[n * W] * m_W_rl[n * W] + m_W_im[n * W] * m_W_im[n * W];	// c^2+d^2
				double part1 = m_X2_rl[index] * m_W_rl[n * W] + m_X2_im[index] * m_W_im[n * W]; // a*c+b*d
				double part2 = m_X2_im[index] * m_W_rl[n * W] - m_X2_rl[index] * m_W_im[n * W]; // b*c-a*d
				if (square > 0)
				{
					m_X2_rl[index] = part1 / square;
					m_X2_im[index] = part2 / square;
				}
			}
		}
		memcpy(m_X_rl, m_X2_rl, sizeof(double) * m_n_fft);
		memcpy(m_X_im, m_X2_im, sizeof(double) * m_n_fft);
	}

	// λ�뵹��
	int index = 0;
	for (int i = 0; i < m_n_fft; i++)
	{
		index = 0;
		for (int m = r - 1; m >= 0; m--)
		{
			index += (1 && (i & (1 << m))) << (r - m - 1);
		}
		m_out_sequence[i].rl = m_X_rl[index];
		m_out_sequence[i].im = m_X_im[index];
		//sprintf(msg + 6, "m_X_rl[i]: %lf, %lf,  index: %d", out_rl[i], out_im[i], index);
		//OutputDebugStringA(msg);
	}

	return true;
}
