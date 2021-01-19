#ifndef _ASR_FEATURE_H_
#define _ASR_FEATURE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

namespace ASR
{
    float hz_to_mel(float freq);
    float mel_to_hz(float mel);

    void show_mat_int(cv::Mat feature_mat, int rows, int cols);
    void show_mat_uchar(cv::Mat feature_mat, int rows, int cols);
    void show_mat_float(cv::Mat feature_mat, int rows, int cols);

    void serial_multiplication(cv::Mat matrix_a, cv::Mat matrix_b, cv::Mat *result);

    void get_mel_filter(cv::Mat *mel_filter, int n_fft = 256, int sample_rate = 8000, int n_mel = 48, int nfilt = 64);
    void get_frequency_feature(short *pdata, int data_len_samples, cv::Mat *frequency_feature, int n_fft = 256, int sample_rate = 8000, int time_step_ms = 10);
    void get_mfsc(cv::Mat &frequency_feature, cv::Mat &mel_filter, cv::Mat *MFSC, int n_mel = 48);
    void get_pcen(cv::Mat &pcen_feature, bool mask = false);
    void get_int_feature(cv::Mat &input, cv::Mat *output, int scale_num = 10);

    struct Feature_Options_S
	{
		Feature_Options_S(): 
            sample_rate(16000), 	
            n_fft(512),
            time_seg_ms(32),
            time_step_ms(10),
            feature_freq(48),
            feature_time(296),
            feature_channels(1),
            pcen_flag(false),
            nfilt(64),
            scale_num(10) {}

        Feature_Options_S(const Feature_Options_S &feature_options): 
            sample_rate(feature_options.sample_rate), 	
            n_fft(feature_options.n_fft),
            time_seg_ms(feature_options.time_seg_ms),
            time_step_ms(feature_options.time_step_ms),
            feature_freq(feature_options.feature_freq),
            feature_time(feature_options.feature_time),
            feature_channels(feature_options.feature_channels),
            pcen_flag(feature_options.pcen_flag),
            nfilt(feature_options.nfilt),
            scale_num(feature_options.sample_rate) {}

		int sample_rate;
        int n_fft;
		int time_seg_ms;
		int time_step_ms;
        int feature_freq;
        int feature_time;
        int feature_channels;
        bool pcen_flag;

        int nfilt;      // get_mel_filter
        int scale_num;  // get_int_feature
	};

    class Feature
    {
    public:
		Feature();
		Feature(const Feature_Options_S &feature_options);
		~Feature();

        inline int feature_time() const { return m_feature_options.feature_time; } 
        inline int feature_freq() const { return m_feature_options.feature_freq; }
        inline int pcen_flag() const { return m_feature_options.pcen_flag; } 
        inline cv::Mat output_feature() const { return m_output_feature; }

	public:
        int check_feature_time(int data_len_samples);
        void copy_mfsc_feature_int_to(unsigned char *feature_data);
        void get_mel_int_feature(short *pdata, int data_len_samples);
        void get_mel_pcen_feature(short *pdata, int data_len_samples);
    
    private:
        void mel_filter_init();

    private:
        Feature_Options_S m_feature_options;

        cv::Mat m_mel_filter;
        cv::Mat m_frequency_feature;
        cv::Mat m_mfsc_feature;
        cv::Mat m_output_feature;
    };
} // namespace ASR

#endif // _ASR_FEATURE_H_