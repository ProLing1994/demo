#include <algorithm>
#include <fstream> 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <fftw3.h>

// const int N=100;
// typedef double arr[2];

#define REAL 0
#define IMAG 1

using namespace std;

int main(int argc, char **argv) {
    std::cout << "Hello World !!" << std::endl;

    // ofstream of1("/home/junwu/Desktop/FFTWTest/in.txt");
    // ofstream of2("/home/junwu/Desktop/FFTWTest/out.txt");

    // of1.precision(6);
    // of2.precision(6);

    // fftw_complex *in, *out;
    // fftw_plan p;
    
    // in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    // out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    
    // for(int i=0;i<N;i++)
    // {
    //     //of1<<sin(i)+sin(10*i)+sin(100*i)<<","<<0.0<<endl;
    //     in[i][0]=sin(i)+sin(10*i)+sin(100*i);
    //     in[i][1]=0;

    // }

    // p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // fftw_execute(p); /* repeat as needed */

    // for(int i=0;i<N;i++)
    // {
    //     of1<<in[i][0]<<","<<in[i][1]<<endl;

    //     of2<<out[i][0]<<","<<out[i][1]<<endl;
    // }
    // fftw_destroy_plan(p);
    // fftw_free(in); fftw_free(out);

	/*
	*fftw_complex 是FFTW自定义的复数类
	*引入<complex>则会使用STL的复数类
	*/
	fftw_complex x[5];
	fftw_complex y[5];

	for (int i = 0; i < 5; i++) {
		x[i][REAL] = i;
		x[i][IMAG] = 0;
	}
	
	//定义plan，包含序列长度、输入序列、输出序列、变换方向、变换模式
	fftw_plan plan = fftw_plan_dft_1d(5, x, y, FFTW_FORWARD, FFTW_ESTIMATE);

	//对于每个plan，应当"一次定义 多次使用"，同一plan的运算速度极快
	fftw_execute(plan);

	for (int i = 0; i < 5; i++) {
		cout << y[i][REAL] << "  " << y[i][IMAG] << endl;
	}

	//销毁plan
	fftw_destroy_plan(plan);
    return 0;
}