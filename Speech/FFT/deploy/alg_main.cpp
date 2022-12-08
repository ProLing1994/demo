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
#include "./common/asr_config.h"

#ifdef _TESTCPU
#include <unistd.h>
#include "cpu_occupy.hpp"
#endif

static ASR::Logger logger = ASR::getLogger();

int main(int argc, char **argv) {
    std::string mount_folder = "/home/huanyuan/share/";
    std::string audio_folder = mount_folder + "audio_data/weakup_asr/weakup_bwc_asr_english/";
    std::string output_folder = mount_folder + "audio_data/weakup_asr/weakup_bwc_asr_english/amba_test";
    std::string model_path = mount_folder + "amba/KWS_model";

    // param init 
    char config_file[256] = {0};
	sprintf(config_file, "%s/kws/configFiles/configFileASR.cfg", model_path.c_str());
	rmCConfig cconfig;
	int rett = cconfig.OpenFile(config_file);
	if (rett != 0)
	{
		printf("[ERROR:] %s, %d: Read Config Failed.\n", __FUNCTION__, __LINE__);
		return rett;
	}
    float receive_audio_length_s = cconfig.GetFloat("NormalParam", "receive_audio_length_s");

    int window_size_ms = int(receive_audio_length_s * 1000);
    int window_stride_ms = int(receive_audio_length_s * 1000);
    int sample_rate = cconfig.GetInt("NormalParam", "fs");
    int window_size_samples = int(sample_rate * window_size_ms / 1000);
    int window_stride_samples = int(sample_rate * window_stride_ms / 1000);
    short audio_data[window_size_samples];
    for (int i = 0; i < window_size_samples; i++) { audio_data[i] = 0;}

    // init
    std::cout << "\033[0;31m" << "[Information:] Init: \033[0;39m" << std::endl;
    int ret = RMAPI_AI_AsrInit(const_cast<char *>(model_path.c_str()));
	if (ret != 0)
	{
        std::cout << "\033[0;31m" << "[ERROR:] RMAPI AI AsrInit Failed." << "\033[0;39m" << std::endl;
		return ret;
	}

    // find auido names
    std::vector<std::string> audio_names;
    ASR::ListDirectorySuffix(audio_folder.c_str(), audio_names);

    // std::vector<std::string> subfolder;
    // std::vector<std::string> audio_names;
    // ASR::ListDirectory(audio_folder.c_str(), subfolder, audio_names);
    
    sort(audio_names.begin(),audio_names.end());

    std::cout << "\033[0;31m" << "[Information:] audio_folder: " << audio_folder << "\033[0;39m" << std::endl;
    std::cout << "\033[0;31m" << "[Information:] audio_names.size(): " << audio_names.size() << "\033[0;39m" << std::endl;
    
    // testcpu
    #ifdef _TESTCPU
    unsigned int pid = ASR::get_pid("rmai_kws_test");
    while (1) {
    #endif

    for (unsigned int idx = 0; idx < audio_names.size(); idx++) {
        std::cout << "\033[0;31m" << "[Information:] audio_names: " << audio_names[idx] << "\033[0;39m" << std::endl;

        // testcpu
        #ifdef _TESTCPU
        ASR::get_proc_cpu_start(pid);
        #endif

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
        char outKeyword[1000] = "";
        int windows_times = int((wave_data.data_length() - window_size_samples) * 1.0 / window_stride_samples) + 1;
        for(int times = 0; times < windows_times; times ++) {
            std::cout << "\033[0;31m" << "[Information:] Audio data stream: " << times * window_stride_samples << " - " << (times + 1) * window_stride_samples \
                        << ", length: " << window_stride_samples <<  "\033[0;39m" << std::endl;

            for(int i = 0; i < window_size_samples; i ++) {
                // audio_data[i] = wave_data.get(i + times * window_stride_samples + 1);
                audio_data[i] = wave_data.get(i + times * window_stride_samples);
            }

            // // check, / 32768.0
            // std::cout << "\033[0;31m" << "[Information:] Audio Data: " << "\033[0;39m" << std::endl;
            // for(unsigned int i = 0; i < window_size_samples; i++) {
            //     std::cout << audio_data[i]<< " ";
            // }
            // std::cout << std::endl;

            // forward
            char out_keyword[100] = "";
            RMAPI_AI_AsrAlgStart(audio_data, window_size_samples, out_keyword);
            sprintf(outKeyword, "%s%s", outKeyword, out_keyword);

            // testcpu
            #ifdef _TESTCPU
            sleep(1);
            char cpuinfo[256];
            sprintf(cpuinfo,"pid: %d, pcpu: %.2f%, procmem: %d KB, virtualmem: %d KB.",  \
                                            pid, ASR::get_proc_cpu_end(pid), \
                                            ASR::get_proc_mem(pid), ASR::get_proc_virtualmem(pid) );
            logger.debug( cpuinfo );
            #endif
        }
        std::cout << "\033[0;31m" << "[Information:] outKeyword: " << outKeyword << "\033[0;39m" << std::endl;
    }

    // testcpu
    #ifdef _TESTCPU
    }
    #endif


    ret = RMAPI_AI_AsrDeinit();

    return 0;
}