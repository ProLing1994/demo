#ifndef _ASR_FEATURE_H_
#define _ASR_FEATURE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#ifdef __cplusplus
extern "C" {
#endif

namespace ASR
{
    float HzToMel(float freq);
	float MelToHz(float mel);
    void ShowMatInt(cv::Mat feature_mat, int rows, int cols);
	void ShowMatUchar(cv::Mat feature_mat, int rows, int cols);
	void ShowMatFloat(cv::Mat feature_mat, int rows, int cols);
    void SerialMultiplication(cv::Mat matrix_a, cv::Mat matrix_b, cv::Mat *result);

    void GetFrequencyFeature(short *pdata, int data_len, cv::Mat *frequency_feature, int n_fft = 256, int sample_rate = 8000, int time_step_ms = 10);
    void GetMelFilter(cv::Mat *mel_filter, int n_fft = 256, int sample_rate = 8000, int n_mel = 48, int nfilt = 64);
    void GetMFSC(cv::Mat &frequency_feature, cv::Mat &mel_filter, cv::Mat *MFSC, int n_mel = 48);
    void GetIntFeat(cv::Mat &input, cv::Mat *output, int scale_num = 10);

    int GetFeatureTime(int data_len, int sample_rate = 8000, int time_seg_ms = 32, int time_step_ms = 10);
    void GetMelIntFeature(short *pdata, int data_len, unsigned char *feature_data, int n_fft = 256, int sample_rate = 8000, int time_seg_ms = 32, int time_step_ms = 10, int feature_freq = 48, int nfilt = 64, int scale_num = 10);
} // namespace ASR

#ifdef __cplusplus
}
#endif

#endif // _ASR_FEATURE_H_