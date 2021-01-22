#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "./common/rm_ASR.hpp"
#include "./common/common.hpp"
#include "./common/wave_data.hpp"

int main(int argc, char **argv) {
    std::string audio_folder = "/home/huanyuan/share/audio_data";
    std::string model_path = "/home/huanyuan/share/KWS_model";

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
    
    for (unsigned int idx = 0; idx < audio_names.size(); idx++) {
        std::string audio_path = audio_folder + "/" + audio_names[idx];
        std::cout << "\033[0;31m" << "[Information:] Audio path: " << audio_path << "\033[0;39m" << std::endl;
        
        // load wav
        ASR::Wave_Data wave_data;
        int ret = wave_data.load_data(audio_path.c_str());
        if(ret == -1) 
        {
            std::cout << "\033[0;31m" << "[ERROR:] Read wav failed." << "\033[0;39m" << std::endl;
            continue;
        }
        if (wave_data.data_length() < window_size_samples) {
            continue;
        }

        std::cout << "\033[0;31m" << "[Information:] Audio Length: " << wave_data.data_length() << ", Audio Fs: " << wave_data.fs() << "\033[0;39m" << std::endl;

        // forward
        int windows_times = int((wave_data.data_length() - window_size_samples) * 1.0 / window_stride_samples) + 1;
        for(int times = 0; times < windows_times; times ++) {
            std::cout << "\033[0;31m" << "\n[Information:] Wave start time: " << times * window_stride_samples << "\033[0;39m" << std::endl;

            for(int i = 0; i < window_size_samples; i ++) {
                audio_data[i] = static_cast<short>(wave_data.data()[i + times * window_stride_samples]);
            }

            // // check, / 32768.0
            // std::cout << "\033[0;31m" << "[Information:] Audio Data: " << "\033[0;39m" << std::endl;
            // for(unsigned int i = 0; i < 10; i ++) {
            //     std::cout << audio_data[i]<< " ";
            // }
            // std::cout << std::endl;

            char outKeyword[100] = "";
            RMAPI_AI_AsrAlgStart(audio_data, window_size_samples, outKeyword);
            std::cout << "\033[0;31m" << "[Information:] outKeyword: " << outKeyword << "\033[0;39m" << std::endl;
        }
    }
    ret = RMAPI_AI_AsrDeinit();

    return 0;
}