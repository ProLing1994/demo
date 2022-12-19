#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "common/wave_data.hpp"
#include "common/feature.hpp"

#include "common/utils/csrc/file_system.h"

int main(int argc, char **argv) {
    // std::string audio_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景";
    // std::string audio_folder = "/home/huanyuan/share/audio_data/english_wav";
    std::string audio_folder = "/home/huanyuan/share/audio_data";
    std::string model_path = "/home/huanyuan/share/novt/KWS_model";

    // init
    int window_len = 512;
    int feature_freq = 48;
    int time_seg_ms = 32;
    int time_step_ms = 10;

    int window_size_ms = 3000;
    int window_stride_ms = 2000;
    int sample_rate = 16000;
    int window_size_samples = int(sample_rate * window_size_ms / 1000);
    int window_stride_samples = int(sample_rate * window_stride_ms / 1000);

    int feature_time = (window_size_samples * 1.0 / sample_rate * 1000 - time_seg_ms) / time_step_ms;
    cv::Mat mel_filter = cv::Mat::zeros(window_len / 2, feature_freq, CV_32FC1);
    cv::Mat speech_feature = cv::Mat::zeros(feature_time, window_len / 2, CV_32FC1);
	cv::Mat mfsc_feature = cv::Mat::zeros(feature_time, feature_freq, CV_32FC1);
	cv::Mat mfsc_feature_int = cv::Mat::zeros(feature_time, feature_freq, CV_8UC1);
	// ASR::get_mel_filter(mel_filter, window_len, sample_rate, feature_freq, 48);
	ASR::get_mel_filter(&mel_filter, window_len, sample_rate, feature_freq);

    short audio_data[window_size_samples] = {0};

    // find auido names
    std::vector<std::string> subfolder;
    std::vector<std::string> audio_names;
    yh_common::list_directory(audio_folder.c_str(), subfolder, audio_names);
    sort(audio_names.begin(),audio_names.end());
    
    for (unsigned int idx = 0; idx < audio_names.size(); idx++) {
        // check 
        if (audio_names[idx].find(".wav") == audio_names[idx].npos)
            continue;

        std::string audio_path = audio_folder + "/" + audio_names[idx];
        std::cout << "\033[0;31m" << "[Information:] Audio path: " << audio_path << "\033[0;39m" << std::endl;
        
        // load wav
        ASR::Wave_Data wave_data;
        wave_data.load_data(audio_path.c_str());
        std::cout << "\033[0;31m" << "[Information:] Audio Length: " << wave_data.data_length() << ", Audio Fs: " << wave_data.fs() << "\033[0;39m" << std::endl;

        // forward
        int windows_times = int((wave_data.data_length() - window_size_samples) * 1.0 / window_stride_samples) + 1;
        for(int times = 0; times < windows_times; times ++) {
            std::cout << "\033[0;31m" << "\n[Information:] Wave start time: " << times * window_stride_samples << "\033[0;39m" << std::endl;

            for(int i = 0; i < window_size_samples; i ++) {
                audio_data[i] = static_cast<short>(wave_data.data()[i + times * window_stride_samples]);
            }

            // check, / 32768.0
            std::cout << "\033[0;31m" << "[Information:] Audio Data: " << "\033[0;39m" << std::endl;
            for(unsigned int i = 0; i < 10; i ++) {
                std::cout << audio_data[i]<< " ";
            }
            std::cout << std::endl;

            ASR::get_frequency_feature(audio_data, window_size_samples, &speech_feature, window_len, sample_rate, time_step_ms);
	        ASR::get_mfsc_feature(speech_feature, mel_filter, &mfsc_feature, feature_freq);

            cv::log(mfsc_feature + 1, mfsc_feature);
            ASR::get_int_feature(mfsc_feature, &mfsc_feature_int);

            std::cout << "\033[0;31m" << "[Information:] mfsc_feature_int.rows: " << mfsc_feature_int.rows << ", mfsc_feature_int.cols: " << mfsc_feature_int.cols <<"\033[0;39m" << std::endl;
            ASR::show_mat_uchar(mfsc_feature_int, 296, 48);  
        }
    }

    return 0;
}