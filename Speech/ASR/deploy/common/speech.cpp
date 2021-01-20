#include <iostream>

#include "common.hpp"
#include "config.h"
#include "speech.hpp"

namespace ASR
{
    Speech_Engine::Speech_Engine()
    {
        m_feature.reset(new Feature);
        m_decode.reset(new Decode);
        
        // TO Do
        // model and image init
    }

    Speech_Engine::Speech_Engine(const ASR_PARAM_S &params, const char *file_path)
    {
        load_param(params, file_path);

        // TO Do
        // model and image init
    }

    Speech_Engine::~Speech_Engine()
    {
        m_feature.reset(nullptr);
        m_decode.reset(nullptr);
        
        // TO Do
        // model and image release
    }
 
    void Speech_Engine::load_param(const ASR_PARAM_S &params, const char *file_path)
    {
        rmCConfig cconfig;
        std::string s, tmp;
        std::ifstream infile;
        
        printf("[Information:] Load param Start!!!\n");

        // check 
        if (file_path == NULL)
        {
            std::cerr << "[ERROR:] Load Param, miss file path" << std::endl;
            return;
        }

        char config_file[256] = {0};
        sprintf(config_file, "%s/configFiles/configFileASR.cfg", file_path);
        cconfig.OpenFile(config_file);

        if (params.algorithem_id == 0)
        {
            printf("[Information:] Initial KWS Param Start!!!\n");

            Feature_Options_S feature_options;
            feature_options.data_len_samples = 48000;
            feature_options.feature_channels = 1;
            feature_options.n_fft = params.n_fft;
            feature_options.feature_freq = params.feature_freq;
            feature_options.feature_time = params.feature_time;
            feature_options.sample_rate = params.sample_rate;
            feature_options.time_seg_ms = params.time_seg_ms;
            feature_options.time_step_ms = params.time_step_ms;
            feature_options.pcen_flag = false;

            m_algorithem_id = params.algorithem_id; 
            m_output_num = params.output_num;
            m_feature.reset(new Feature(feature_options));

            std::vector<std::string> keywords;
            std::vector<std::string> symbol_list;
            int kws_num = cconfig.GetInt("KwsParam", "kws_num");
            keywords.push_back("");
            for (int i = 1; i <= kws_num; i++)
            {
                s = std::to_string(i);
                tmp = cconfig.GetStr("CQ_KWS_18", s.c_str());
                keywords.push_back(tmp);
            }

            sprintf(config_file, "%s/configFiles/dict_without_tone.txt", file_path);
            infile.open(config_file);
            assert(infile.is_open());
            while (getline(infile, s))
            {
                tmp = s;
                tmp.erase(tmp.end() - 1);
                symbol_list.push_back(tmp);
            }
            infile.close();

            m_decode.reset(new Decode(keywords, symbol_list));
            printf("[Information:] Initial KWS Param Done!!!\n");
        }
        else
        {
            printf("[Information:] Initial ASR Param Start!!!\n");
            Feature_Options_S feature_options;
            feature_options.data_len_samples = 48000;
            feature_options.feature_channels = 1;
            feature_options.n_fft = params.n_fft;
            feature_options.feature_freq = params.feature_freq;
            feature_options.feature_time = params.feature_time;
            feature_options.sample_rate = params.sample_rate;
            feature_options.time_seg_ms = params.time_seg_ms;
            feature_options.time_step_ms = params.time_step_ms;
            feature_options.pcen_flag = false;

            m_algorithem_id = params.algorithem_id; 
            m_output_num = params.output_num;
            m_fc1_dim = params.fc1_kernels;
            m_fc2_dim = params.fc2_kernels;
            m_feature.reset(new Feature(feature_options));

            std::vector<std::string> symbol_list;
            std::vector<std::string> hanzi_kws_list;
            std::vector<std::vector<std::string>> pinyin_kws_list;
            char config_file[256] = "";
            sprintf(config_file, "%s/configFiles/dict_without_tone.txt", file_path);

            infile.open(config_file);
            assert(infile.is_open());
            while (getline(infile, s))
            {
                tmp = s;
                tmp.erase(tmp.end() - 1);
                symbol_list.push_back(tmp);
            }
            infile.close();

            memset(config_file, '\0', sizeof(config_file));
            sprintf(config_file, "%s/configFiles/configFileASR.cfg", file_path);
            cconfig.OpenFile(config_file);

            int kws_num = cconfig.GetInt("AsrParam", "kws_num");

            for (int i = 1; i <= kws_num; i++)
            {
                std::vector<std::string> tmp_pny;
                s = std::to_string(i);
                tmp = cconfig.GetStr("Demo_Mandarin_KWS", s.c_str());
                hanzi_kws_list.push_back(tmp);
                tmp = cconfig.GetStr("Demo_Mandarin_id", tmp.c_str());
                tmp_pny = ASR::StringSplit(tmp, " ");
                pinyin_kws_list.push_back(tmp_pny);
            }

            m_decode.reset(new Decode(0, symbol_list, hanzi_kws_list, pinyin_kws_list));
            printf("[Information:] Initial ASR Param Done!!!\n");
        }

        printf("[Information:] Load load_param Done!!!\n");
    }

    void Speech_Engine::speech_recognition(short *pdata, int data_len_samples, char *outKeyword)
    {
        //std::cerr<<"====RMAPI_AI_ASR_RUN_Start====="<<std::endl;
        if (m_algorithem_id == 0)
        {
            kws_asr_recognition(pdata, data_len_samples, outKeyword);
        }
        else
        {
            asr_recognition(pdata, data_len_samples, outKeyword);
        }
    }

    void Speech_Engine::asr_recognition(short *pdata, int data_len_samples, char *outKeyword)
    {
        // check
        int ret = m_feature->check_data_length(data_len_samples);
        if (ret != 0)
        {
            std::cerr << "[ERROR:] asr_recognition input data length is not 24000 " << std::endl;
            return;
        }

        m_feature->get_featuer_total_window(pdata, data_len_samples);
        cv::Mat output_feature = m_feature->mfsc_feature_int();

        std::cout << "\033[0;31m" << "[Information:] output_feature.rows: " << output_feature.rows << ", output_feature.cols: " << output_feature.cols <<"\033[0;39m" << std::endl;
		ASR::show_mat_uchar(output_feature, 296, 48);

        return;
    }

    void Speech_Engine::kws_asr_recognition(short *pdata, int data_len_samples, char *outKeyword)
    {
        // check
        int ret = m_feature->check_data_length(data_len_samples);
        if (ret != 0)
        {
            std::cerr << "[ERROR:] asr_recognition input data length is not 24000 " << std::endl;
            return;
        }

        // get feature
        m_feature->get_featuer_slides_window(pdata, data_len_samples, 48);

        for (int i = 0; i < m_feature->data_mat_time() - m_feature->feature_time(); i = i + m_feature->time_step_ms())
        {
            m_feature->get_single_feature(i);
            cv::Mat output_feature = m_feature->single_feature();

            std::cout << "\033[0;31m" << "[Information:] output_feature.rows: " << output_feature.rows << ", output_feature.cols: " << output_feature.cols <<"\033[0;39m" << std::endl;
            ASR::show_mat_uchar(output_feature, 120, 48);
        }
    }

} // namespace ASR