#include <algorithm>

#include "feature.hpp"
#include "fft.h"

namespace ASR
{
    float hz_to_mel(float freq)
    {
        float b;
        b = 2595 * log10(1 + freq / 700.0);
        return b;
    }

    float mel_to_hz(float mel)
    {
        float b;
        b = 700 * (pow(10, (mel / 2595.0)) - 1);
        return b;
    }

    void show_mat_int(cv::Mat feature_mat, int rows, int cols)
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

    void show_mat_uchar(cv::Mat feature_mat, int rows, int cols)
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

    void show_mat_float(cv::Mat feature_mat, int rows, int cols)
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

    void serial_multiplication(cv::Mat matrix_a, cv::Mat matrix_b, cv::Mat *result)
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

    void get_mel_filter(cv::Mat *mel_filter, int n_fft, int sample_rate, int n_mel, int nfilt)
    {
        int low_freq = 10;
        int high_freq = sample_rate / 2;
        int samplerate = sample_rate;
        int nfft = n_fft;
        if (high_freq > samplerate / 2)
        {
            high_freq = samplerate / 2;
        }
        float low_mel = hz_to_mel(low_freq);
        float high_mel = hz_to_mel(high_freq);
        float *bin = new float[nfilt + 2];
        float *melpoints = new float[nfilt + 2];
        for (int i = 0; i < nfilt + 2; i++)
        {
            melpoints[i] = (high_mel - low_mel) * i / (nfilt + 1) + low_mel;
            bin[i] = floor((nfft + 1) * mel_to_hz(melpoints[i]) / samplerate);
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

    void get_mfsc_feature(cv::Mat &frequency_feature, cv::Mat &mel_filter, cv::Mat *MFSC, int n_mel)
    {
        (void) n_mel;
        serial_multiplication(frequency_feature, mel_filter, MFSC);
        return;
    }

    void get_frequency_feature(short *pdata, int data_len_samples, cv::Mat *frequency_feature, int n_fft, int sample_rate, int time_step_ms)
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

        int range0_end = std::min(16000.0, (data_len_samples * 1.0 / sample_rate * 1000 - n_fft * 1000 / sample_rate) / time_step_ms); // 10 #

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

    void get_pcen_feature(cv::Mat &pcen_feature, bool mask)
    {
        float s = 0.025;
        float m_s = 1 - s;
        // float a = 1.00;
        float o = 2;
        float r = 0.5;
        float e = 0.00001;
        float o_r = pow(o, r);
        int freq_num = pcen_feature.cols;
        int frame_num = pcen_feature.rows;
        int channel = pcen_feature.channels();
        mask = false;
        cv::Mat M;
        cv::Mat PCEN;
        if (channel == 3)
        {
            M = cv::Mat::zeros(freq_num, 1, CV_32FC3);
            PCEN = cv::Mat::zeros(frame_num, freq_num, CV_32FC3);
        }
        else
        {
            M = cv::Mat::zeros(freq_num, 1, CV_32FC1);
            PCEN = cv::Mat::zeros(frame_num, freq_num, CV_32FC1);
        }
        float *temp_m = new float[channel]();
        for (int i = 0; i < freq_num; i++)
        {
            for (int k = 0; k < channel; k++)
            {
                temp_m[k] = 0;
                for (int j = 0; j < 20; j++)
                {
                    temp_m[k] += pcen_feature.at<float>(j, i, k);
                }
                M.at<float>(i, 0, k) = temp_m[k] / 20;
            }
        }
        delete[] temp_m;

        if (mask)
        {
            // TO Do
        }
        else
        {
            for (int i = 0; i < frame_num; i++)
            {
                for (int j = 0; j < freq_num; j++)
                {
                    for (int k = 0; k < channel; k++)
                    {
                        M.at<float>(j, 0, k) = m_s * M.at<float>(j, 0, k) + s * pcen_feature.at<float>(i, j, k);
                        PCEN.at<float>(i, j, k) = pow((pcen_feature.at<float>(i, j, k) / (e + M.at<float>(j, 0, k) + 10.0) + o), r) - o_r;
                    }
                }
            }
        }
        pcen_feature = PCEN.clone();
    }

    void get_int_feature(cv::Mat &input, cv::Mat *output, int scale_num)
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

    Feature::Feature()
    {
        m_feature_options = Feature_Options_S();        
        feature_mat_init();
        mel_filter_init();
    }

    Feature::Feature(const Feature_Options_S &feature_options)
    {
        m_feature_options = Feature_Options_S(feature_options);
        feature_mat_init();
        mel_filter_init();
    }

    Feature::~Feature()
    {

    }

    void Feature::mel_filter_init()
    {
        // init 
        get_mel_filter(&m_mel_filter64, m_feature_options.n_fft, m_feature_options.sample_rate, m_feature_options.feature_freq);
        get_mel_filter(&m_mel_filter48, m_feature_options.n_fft, m_feature_options.sample_rate, m_feature_options.feature_freq, 48);
    }

    void Feature::feature_mat_init()
    {   
        m_mel_filter48 = cv::Mat::zeros(m_feature_options.n_fft / 2, m_feature_options.feature_freq, CV_32FC1);
        m_mel_filter64 = cv::Mat::zeros(m_feature_options.n_fft / 2, m_feature_options.feature_freq, CV_32FC1);
        m_frequency_feature = cv::Mat::zeros(m_feature_options.data_mat_time, m_feature_options.n_fft / 2, CV_32FC1);
        m_mfsc_feature = cv::Mat::zeros(m_feature_options.data_mat_time, m_feature_options.feature_freq, CV_32FC1);
        m_mfsc_feature_int = cv::Mat::zeros(m_feature_options.data_mat_time, m_feature_options.feature_freq, CV_8UC1);

        m_single_feature = cv::Mat::zeros(m_feature_options.feature_time, m_feature_options.feature_freq, CV_8UC1);
    }

    int Feature::check_data_length(int data_len_samples)
    {
        if(data_len_samples == m_feature_options.data_len_samples)
            return 0;
        else
            return -1;
    }

    void Feature::copy_mfsc_feature_int_to(unsigned char *feature_data)
    {
        memcpy(feature_data, m_mfsc_feature_int.data, m_feature_options.feature_time * m_feature_options.feature_freq * sizeof(unsigned char));
    }

    void Feature::get_mfsc_feature_filter(int mel_filter)
    {
        if(mel_filter == 64)
            get_mfsc_feature(m_frequency_feature, m_mel_filter64, &m_mfsc_feature, m_feature_options.feature_freq);
        else if(mel_filter == 48)
            get_mfsc_feature(m_frequency_feature, m_mel_filter48, &m_mfsc_feature, m_feature_options.feature_freq);
        else
            std::cerr << "[ERROR:] unknow mel_filter, please check!!! " << std::endl;
    }

    void Feature::get_mel_int_feature(short *pdata, int data_len_samples, int mel_filter)
    {
        get_frequency_feature(pdata, data_len_samples, &m_frequency_feature, m_feature_options.n_fft, m_feature_options.sample_rate, m_feature_options.time_step_ms);
        get_mfsc_feature_filter(mel_filter);
        cv::log(m_mfsc_feature + 1, m_mfsc_feature);
        get_int_feature(m_mfsc_feature, &m_mfsc_feature_int);
    }

    void Feature::get_mel_pcen_feature(short *pdata, int data_len_samples, int mel_filter)
    {
        int u8_data = 0;

        get_frequency_feature(pdata, data_len_samples, &m_frequency_feature, m_feature_options.n_fft, m_feature_options.sample_rate, m_feature_options.time_step_ms);
        get_mfsc_feature_filter(mel_filter);
		get_pcen_feature(m_mfsc_feature);
		for (int r = 0; r < m_feature_options.feature_time; r++)
		{
			for (int c = 0; c < m_feature_options.feature_freq; c++)
			{
				for (int k = 0; k < m_feature_options.feature_channels; k++)
				{
					u8_data = (int)(m_mfsc_feature.at<float>(r, c, k) * 255 / 3);
					u8_data = u8_data < 0 ? 0 : u8_data;
					u8_data = u8_data > 255 ? 255 : u8_data;
					m_mfsc_feature_int.at<unsigned char>(r, c, k) = u8_data;
				}
			}
		}
    }

    void Feature::get_featuer_total_window(short *pdata, int data_len_samples)
    {
        if(m_feature_options.pcen_flag == true)
        {
            get_mel_pcen_feature(pdata, data_len_samples);
        }
        else
        {
            get_mel_int_feature(pdata, data_len_samples);
        }
    }

    void Feature::get_featuer_slides_window(short *pdata, int data_len_samples, int mel_filter)
    {
        if(m_feature_options.pcen_flag == true)
        {
            // do FFT to all wav, then do PCEN to each fragment
            get_frequency_feature(pdata, data_len_samples, &m_frequency_feature, m_feature_options.n_fft, m_feature_options.sample_rate, m_feature_options.time_step_ms);
            get_mfsc_feature_filter(mel_filter);
        }
        else
        {
            get_mel_int_feature(pdata, data_len_samples, mel_filter);
        }
    }
    
    void Feature::get_single_feature(int start_feature_time)
    {
        if (m_feature_options.pcen_flag == true)
        {
            cv::Mat pcen_feature = m_mfsc_feature(cv::Range(start_feature_time, start_feature_time + m_feature_options.feature_time), cv::Range(0, m_feature_options.feature_freq)).clone();
            get_pcen_feature(pcen_feature);
            get_int_feature(pcen_feature, &m_single_feature, 3);
        }
        else
        {
            m_single_feature = m_mfsc_feature_int(cv::Range(start_feature_time, start_feature_time + m_feature_options.feature_time), cv::Range(0, m_feature_options.feature_freq)).clone();
        }
    }
} // namespace ASR