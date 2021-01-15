#include <algorithm>

#include "feature.hpp"
#include "fft.h"

namespace ASR
{
    float HzToMel(float freq)
    {
        float b;
        b = 2595 * log10(1 + freq / 700.0);
        return b;
    }

    float MelToHz(float mel)
    {
        float b;
        b = 700 * (pow(10, (mel / 2595.0)) - 1);
        return b;
    }

    void ShowMatInt(cv::Mat feature_mat, int rows, int cols)
    {
        int *p_data = (int *)feature_mat.data;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cerr << (int)*(p_data + j + i * feature_mat.cols) << " ";
            }
            std::cerr << std::endl;
        }
    }

    void ShowMatUchar(cv::Mat feature_mat, int rows, int cols)
    {
        unsigned char *p_data = (unsigned char *)feature_mat.data;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cerr << (int)*(p_data + j + i * feature_mat.cols) << " ";
            }
            std::cerr << std::endl;
        }
    }

    void ShowMatFloat(cv::Mat feature_mat, int rows, int cols)
    {
        float *p_data = (float *)feature_mat.data;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cerr << (float)*(p_data + j + i * feature_mat.cols) << " ";
            }
            std::cerr << std::endl;
        }
    }

    void SerialMultiplication(cv::Mat matrix_a, cv::Mat matrix_b, cv::Mat *result)
    {
        int M = matrix_a.rows;
        int N = matrix_a.cols;
        int O = matrix_b.cols;
        float *p_a = (float *)matrix_a.data;
        float *p_b = (float *)matrix_b.data;
        float *p_o = (float *)result->data;

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < O; ++j)
            {
                double temp = 0;
                for (int k = 0; k < N; ++k)
                {
                    temp += *(p_a + i * N + k) * *(p_b + k * O + j);
                }
                *(p_o + i * O + j) = temp;
            }
        }
    }

    void GetMelFilter(cv::Mat *mel_filter, int n_fft, int sample_rate, int n_mel, int nfilt)
    {
        int low_freq = 10;
        int high_freq = sample_rate / 2;
        int samplerate = sample_rate;
        int nfft = n_fft;
        if (high_freq > samplerate / 2)
        {
            high_freq = samplerate / 2;
        }
        float low_mel = HzToMel(low_freq);
        float high_mel = HzToMel(high_freq);
        float *bin = new float[nfilt + 2];
        float *melpoints = new float[nfilt + 2];
        for (int i = 0; i < nfilt + 2; i++)
        {
            melpoints[i] = (high_mel - low_mel) * i / (nfilt + 1) + low_mel;
            bin[i] = floor((nfft + 1) * MelToHz(melpoints[i]) / samplerate);
        }
        for (int j = 0; j < n_mel; j++)
        {
            for (int i = bin[j]; i < bin[j + 1]; i++)
            {
                mel_filter->at<float>(i, j) = (i - bin[j]) / (bin[j + 1] - bin[j]);
            }
            for (int i = bin[j + 1]; i < bin[j + 2]; i++)
            {
                mel_filter->at<float>(i, j) = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1]);
            }
        }
        delete[] bin;
        delete[] melpoints;
        return;
    }

    void GetMFSC(cv::Mat &frequency_feature, cv::Mat &mel_filter, cv::Mat *MFSC, int n_mel)
    {
        (void) n_mel;
        SerialMultiplication(frequency_feature, mel_filter, MFSC);
        return;
    }

    void GetFrequencyFeature(short *pdata, int data_len, cv::Mat *frequency_feature, int n_fft, int sample_rate, int time_step_ms)
    {
        // wav time step 10ms
        rm_FFT fft(n_fft);
        bool with_log = false;
        float w[n_fft] = {0};
        float data_line[n_fft] = {0};
        for (int i = 0; i < n_fft; i++)
        {
            w[i] = 0.54 - 0.46 * cos((float)2 * PI * i / (n_fft - 1));
        }

        int range0_end = std::min(16000.0, (data_len * 1.0 / sample_rate * 1000 - n_fft * 1000 / sample_rate) / time_step_ms); // 10 #

        if (!frequency_feature->isContinuous())
            return;
        for (int i = 0; i < range0_end - 1; i++)
        {

            int p_start = i * sample_rate * time_step_ms / 1000;
            for (int j = 0; j < n_fft; j++)
            {
                data_line[j] = *(pdata + p_start + j) * w[j];
            }

            int len = sizeof(data_line) / sizeof(float);
            memset(fft.inVec, 0, len * sizeof(Complex));
            for (int j = 0; j < len; j++)
                fft.inVec[j].rl = data_line[j];
            /*
            if (n_fft != FFT_N)
            {
                printf("window length does not match FFT length!!!\n");
                return;
            }
            */
            fft.fft(fft.inVec, len, fft.outVec);
            if (with_log)
            {
                for (int j = 0; j < len / 2; j++)
                {
                    frequency_feature->at<float>(i, j) = log((sqrt(pow(fft.outVec[j].rl, 2) + pow(fft.outVec[j].im, 2)) / n_fft) + 1);
                }
            }
            else
            {
                for (int j = 0; j < len / 2; j++)
                {
                    frequency_feature->at<float>(i, j) = sqrt(pow(fft.outVec[j].rl, 2) + pow(fft.outVec[j].im, 2)) / n_fft;
                }
            }
        }
        return;
    }

    void GetIntFeat(cv::Mat &input, cv::Mat *output, int scale_num)
    {
        int u8_data = 0;
        for (int r = 0; r < input.rows; r++)
        {
            for (int c = 0; c < input.cols; c++)
            {
                for (int k = 0; k < input.channels(); k++)
                {
                    u8_data = (int)(input.at<float>(r, c, k) * 255 / scale_num);
                    u8_data = u8_data < 0 ? 0 : u8_data;
                    u8_data = u8_data > 255 ? 255 : u8_data;
                    output->at<unsigned char>(r, c, k) = u8_data;
                }
            }
        }
    }

    int GetFeatureTime(int data_len, int sample_rate, int time_seg_ms, int time_step_ms)
    {
        int feature_time = (data_len * 1.0 / sample_rate * 1000 - time_seg_ms) / time_step_ms;
        return feature_time;
    }

    void GetMelIntFeature(short *pdata, int data_len, unsigned char *feature_data, int n_fft, int sample_rate, int time_seg_ms, int time_step_ms, int feature_freq, int nfilt, int scale_num)
    {
        // init 
        int feature_time = (data_len * 1.0 / sample_rate * 1000 - time_seg_ms) / time_step_ms;
        cv::Mat mel_filter = cv::Mat::zeros(n_fft / 2, feature_freq, CV_32FC1);
        cv::Mat frequency_feature = cv::Mat::zeros(feature_time, n_fft / 2, CV_32FC1);
        cv::Mat mfsc_feature = cv::Mat::zeros(feature_time, feature_freq, CV_32FC1);
        cv::Mat mfsc_feature_int = cv::Mat::zeros(feature_time, feature_freq, CV_8UC1);
        GetMelFilter(&mel_filter, n_fft, sample_rate, feature_freq);

        GetFrequencyFeature(pdata, data_len, &frequency_feature, n_fft, sample_rate, time_step_ms);
        GetMFSC(frequency_feature, mel_filter, &mfsc_feature, feature_freq);

        cv::log(mfsc_feature + 1, mfsc_feature);
        GetIntFeat(mfsc_feature, &mfsc_feature_int);

        // std::cout << "\033[0;31m" << "[Information:] mfsc_feature_int.rows: " << mfsc_feature_int.rows << ", mfsc_feature_int.cols: " << mfsc_feature_int.cols <<"\033[0;39m" << std::endl;
        // ShowMatUchar(mfsc_feature_int, 296, 48);

        memcpy(feature_data, mfsc_feature_int.data, feature_time*feature_freq*sizeof(unsigned char));
    }

} // namespace ASR