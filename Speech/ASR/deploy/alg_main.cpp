#include <algorithm>
#include <fstream> 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "./common/rm_ASR.hpp"
#include "./common/common.hpp"
#include "./common/wave_data.hpp"

#ifdef _TESTTIME
#ifndef TEST_TIME
#include <sys/time.h>
#define TEST_TIME(times) do{\
        struct timeval cur_time;\
        gettimeofday(&cur_time, NULL);\
        times = (cur_time.tv_sec * 1000000llu + cur_time.tv_usec) / 1000llu;\
    }while(0)
#endif

int asr_alg_times = 0;
double asr_alg_time = 0;
extern double feature_time;
extern double model_forward_time;
extern double decode_time;
#endif

int main(int argc, char **argv) {
    // std::string audio_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景";
    // std::string output_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/test";
    // std::string audio_folder = "/home/huanyuan/share/audio_data/english_wav";
    // std::string output_folder = "/home/huanyuan/share/audio_data/english_wav/amba_test";
    std::string audio_folder = "/home/huanyuan/share/audio_data";
    std::string output_folder = "/home/huanyuan/share/audio_data/novt_test";
    std::string model_path = "/home/huanyuan/share/novt/KWS_model";

    // init
    int ret = RMAPI_AI_AsrInit(const_cast<char *>(model_path.c_str()));
	if (ret != 0)
	{
        std::cout << "\033[0;31m" << "[ERROR:] RMAPI AI AsrInit Failed." << "\033[0;39m" << std::endl;
		return ret;
	}

    int window_size_ms = 3000;
    int window_stride_ms = 2000;
    int sample_rate = 16000;
    int window_size_samples = int(sample_rate * window_size_ms / 1000);
    int window_stride_samples = int(sample_rate * window_stride_ms / 1000);

    short audio_data[window_size_samples] = {0};

    // find auido names
    std::vector<std::string> subfolder;
    std::vector<std::string> audio_names;
    ASR::ListDirectory(audio_folder.c_str(), subfolder, audio_names);
    sort(audio_names.begin(),audio_names.end());
    
    // output csv 
    std::ofstream ofile(output_folder + "/result.csv", std::ios::trunc);
    ofile << "data," << "amba_result" << std::endl;

    for (unsigned int idx = 0; idx < audio_names.size(); idx++) {
        // check 
        if (audio_names[idx].find(".wav") == audio_names[idx].npos)
            continue;

        std::string audio_path = audio_folder + "/" + audio_names[idx];
        std::cout << "\033[0;31m" << "[Information:] Audio path: " << audio_path << "\033[0;39m" << std::endl;
        
        // load wav
        ASR::Wave_Data wave_data;
        int ret = wave_data.load_data(audio_path.c_str());
        if(ret == -1) 
        {
            std::cout << "\033[0;31m" << "[ERROR:] Read wave failed." << "\033[0;39m" << std::endl;
            continue;
        }
        if (wave_data.data_length() < window_size_samples) {
            std::cout << "\033[0;31m" << "[ERROR:] Wave time is too short." << "\033[0;39m" << std::endl;
            continue;
        }
        std::cout << "\033[0;31m" << "[Information:] Audio Length: " << wave_data.data_length() << ", Audio Fs: " << wave_data.fs() << "\033[0;39m" << std::endl;

        // forward
        #ifdef _TESTTIME
        unsigned long long time_begin, time_end;
        #endif

        char outKeyword[1000] = "";
        int windows_times = int((wave_data.data_length() - window_size_samples) * 1.0 / window_stride_samples) + 1;
        for(int times = 0; times < windows_times; times ++) {
            std::cout << "\033[0;31m" << "[Information:] Wave start time: " << times * window_stride_samples << "\033[0;39m" << std::endl;

            for(int i = 0; i < window_size_samples; i ++) {
                audio_data[i] = static_cast<short>(wave_data.data()[i + times * window_stride_samples]);
            }

            // check, / 32768.0
            std::cout << "\033[0;31m" << "[Information:] Audio Data: " << "\033[0;39m" << std::endl;
            for(unsigned int i = 0; i < 10; i ++) {
                std::cout << audio_data[i]<< " ";
            }
            std::cout << std::endl;

            // forward
            #ifdef _TESTTIME
            TEST_TIME(time_begin);
            #endif

            RMAPI_AI_AsrAlgStart(audio_data, window_size_samples, outKeyword);

            #ifdef _TESTTIME
            TEST_TIME(time_end);
            asr_alg_times += 1;
            asr_alg_time += time_end - time_begin;
            #endif
        }
        std::cout << "\033[0;31m" << "[Information:] outKeyword: " << outKeyword << "\033[0;39m" << std::endl << std::endl;
        ofile << std::string(audio_names[idx]) << ",";
        ofile << std::string(outKeyword) << std::endl;
    }
    
    #ifdef _TESTTIME
    std::cout << "\033[0;31m" << "[Information:] asr_alg_time: " << asr_alg_time / asr_alg_times << " ms." << "\033[0;39m" << std::endl;
    std::cout << "\033[0;31m" << "[Information:] feature_time: " << feature_time / asr_alg_times << " ms." << "\033[0;39m" << std::endl;
    std::cout << "\033[0;31m" << "[Information:] model_forward_time: " << model_forward_time / asr_alg_times << " ms." << "\033[0;39m" << std::endl;
    std::cout << "\033[0;31m" << "[Information:] decode_time: " << decode_time / asr_alg_times << " ms." << "\033[0;39m" << std::endl;
    #endif

    ofile.close();
    ret = RMAPI_AI_AsrDeinit();

    return 0;
}