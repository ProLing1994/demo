#include <iostream>

#include "common.hpp"
#include "config.h"
#include "speech.hpp"

#ifdef _TESTTIME
#ifndef TEST_TIME
#include <sys/time.h>
#define TEST_TIME(times) do{\
        struct timeval cur_time;\
        gettimeofday(&cur_time, NULL);\
        times = (cur_time.tv_sec * 1000000llu + cur_time.tv_usec) / 1000llu;\
    }while(0)
#endif
double feature_time = 0;
double model_forward_time = 0;
double decode_time = 0;
#endif

namespace ASR
{
    Speech_Engine::Speech_Engine()
    {
        m_feature.reset(new Feature);
        m_decode.reset(new Decode);
        m_model.reset(new Model);
    }

    Speech_Engine::Speech_Engine(const ASR_PARAM_S &params, const char *file_path)
    {
        init_feature_decode_model(params, file_path);
    }

    Speech_Engine::~Speech_Engine()
    {
        m_feature.reset(nullptr);
        m_decode.reset(nullptr);
        m_model.reset(nullptr);
    }
 
    void Speech_Engine::init_feature_decode_model(const ASR_PARAM_S &params, const char *file_path)
    {
        rmCConfig cconfig;
        std::string s, tmp;
        std::ifstream infile;
        
        printf("[Information:] Init Feature && Decode && Model Start.\n");

        // check 
        if (file_path == nullptr)
        {
            printf("[ERROR:] %s, %d: Load param, miss file path.\n", __FUNCTION__, __LINE__);
            return;
        }

        char config_file[256] = {0};
        sprintf(config_file, "%s/configFiles/configFileASR.cfg", file_path);
        cconfig.OpenFile(config_file);

        if (params.algorithem_id == 0)
        {   
            // Params
            m_algorithem_id = params.algorithem_id; 
            m_language_id = params.language_id; 

            // Feature
            printf("[Information:] Init KWS Feature.\n");
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

            m_feature.reset(new Feature(feature_options));

            // Decode
            printf("[Information:] Init KWS Decode.\n");
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

            sprintf(config_file, "%s/configFiles/%s", file_path, params.bpe_path.c_str());
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

            // Model
            printf("[Information:] Init KWS Model.\n");

            Model_Options_S model_options;
            model_options.input_feature_freq = params.feature_freq; 
            model_options.input_feature_time = params.feature_time; 
            model_options.output_feature_num = params.output_num; 
            model_options.output_feature_time = 1; 

            m_model.reset(new Model(model_options));

            // char model_path[128] = "";
            // sprintf(model_path, "%s/%s", file_path, "mandarin_asr_nofc_16K.bin");
            // m_model->asr_init(model_path, m_feature->feature_freq(),  m_feature->feature_time());
        }
        else
        {
            // Params
            m_algorithem_id = params.algorithem_id; 
            m_language_id = params.language_id; 

            // Feature
            printf("[Information:] Init ASR Feature.\n");
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

            m_feature.reset(new Feature(feature_options));

            // Decode
            printf("[Information:] Init ASR Decode.\n");
            std::vector<std::string> symbol_list;
            std::vector<std::string> hanzi_kws_list;
            std::vector<std::vector<std::string>> pinyin_kws_list;
            std::vector<std::vector<std::string>> english_kws_list;
            char config_file[256] = "";
            sprintf(config_file, "%s/configFiles/%s", file_path, params.bpe_path.c_str());

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

            int mandarin_kws_num = cconfig.GetInt("AsrParam", "mandarin_kws_num");
            for (int i = 1; i <= mandarin_kws_num; i++)
            {
                std::vector<std::string> tmp_pny;
                s = std::to_string(i);
                tmp = cconfig.GetStr("Demo_Mandarin_KWS", s.c_str());
                hanzi_kws_list.push_back(tmp);
                tmp = cconfig.GetStr("Demo_Mandarin_id", tmp.c_str());
                tmp_pny = ASR::StringSplit(tmp, " ");
                pinyin_kws_list.push_back(tmp_pny);
            }

            int english_kws_num = cconfig.GetInt("AsrParam", "english_kws_num");
            for (int i = 1; i <= english_kws_num; i++)
            {
                std::vector<std::string> tmp_pny;
                s = std::to_string(i);
                tmp = cconfig.GetStr("Demo_English_KWS", s.c_str());
                tmp_pny = ASR::StringSplit(tmp, " ");
                english_kws_list.push_back(tmp_pny);
            }

            m_decode.reset(new Decode(symbol_list, hanzi_kws_list, pinyin_kws_list, english_kws_list));

            // Model
            printf("[Information:] Init KWS Model.\n");

            Model_Options_S model_options;
            model_options.input_feature_freq = params.feature_freq; 
            model_options.input_feature_time = params.feature_time; 
            model_options.output_feature_num = params.output_num; 
            // If the output shape is uncertain, the parameter is set to zero, output_feature_time = 35; 
            if(params.output_time != 0)
                model_options.output_feature_time = params.output_time; 

            m_model.reset(new Model(model_options));

            char model_path[128] = "";
            sprintf(model_path, "%s/%s", file_path, params.model_name.c_str());
            m_model->asr_init(model_path, m_feature->feature_freq(),  m_feature->feature_time(), params.output_name.c_str());
        }

        printf("[Information:] Init Feature && Decode && Model Done.\n");
    }

    void Speech_Engine::speech_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        //std::cout<<"====RMAPI_AI_ASR_RUN_Start====="<<std::endl;
        if (m_algorithem_id == 0) {
            kws_asr_recognition(pdata, data_len_samples, output_keyword);
        }
        else if (m_algorithem_id == 1){
            asr_recognition(pdata, data_len_samples, output_keyword);
        }
        else {
            printf("[ERROR:] %s, %d: Unknow m_algorithem_id.\n", __FUNCTION__, __LINE__);
            return;
        }

    }

    void Speech_Engine::asr_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        // init 
        std::string output_str;
        cv::Mat cnn_out = cv::Mat::zeros(m_model->output_feature_time(), m_model->output_feature_num(), CV_32SC1);

        // check
        int ret = m_feature->check_data_length(data_len_samples);
        if (ret != 0)
        {
            printf("[ERROR:] %s, %d: Input data length is not 24000.\n", __FUNCTION__, __LINE__);
            return;
        }

        #ifdef _TESTTIME
        unsigned long long time_begin, time_end;
        TEST_TIME(time_begin);
        #endif

        // feature
        m_feature->get_featuer_total_window(pdata, data_len_samples);
        cv::Mat output_feature = m_feature->mfsc_feature_int();
        std::cout << "[Information:] output_feature.rows: " << output_feature.rows << ", output_feature.cols: " << output_feature.cols << std::endl;
		ASR::show_mat_uchar(output_feature, 296, 56);

        #ifdef _TESTTIME
        TEST_TIME(time_end);
        feature_time += time_end - time_begin;
        TEST_TIME(time_begin);
        #endif

        // forward
        ret = m_model->asr_forward(output_feature, &cnn_out);
        if (ret != 0)
        {
            printf("[ERROR:] %s, %d: ASR Forward Failed.\n", __FUNCTION__, __LINE__);
            return;
        }
        // std::cout << "[Information:] cnn_out.rows: " << cnn_out.rows << ", cnn_out.cols: " << cnn_out.cols << std::endl;
        // ASR::show_mat_float(cnn_out, 74, 1000);

        #ifdef _TESTTIME
        TEST_TIME(time_end);
        model_forward_time += time_end - time_begin;
        TEST_TIME(time_begin);
        #endif

        // decode 
        m_decode->ctc_decoder(cnn_out);

        if(m_language_id == 0) {
            // chinese
            // m_decode->show_result_id();
            // m_decode->show_symbol();
            // m_decode->output_symbol(&output_str);
            m_decode->match_keywords_robust(&output_str);
        }
        else if (m_language_id == 1) {
            // english
            // m_decode->show_result_id();
            // m_decode->show_symbol();
            // m_decode->show_symbol_english();
            // m_decode->output_symbol(&output_str);
            // m_decode->output_symbol_english(&output_str);
            m_decode->match_keywords_english(&output_str);
        }
        else {
            printf("[ERROR:] %s, %d: Unknow m_language_id.\n", __FUNCTION__, __LINE__);
            return;
        }

        std::cout << "[Information:] output_str: " << output_str << std::endl;
        sprintf(output_keyword, "%s%s", output_keyword, output_str.c_str());

        #ifdef _TESTTIME
        TEST_TIME(time_end);
        decode_time += time_end - time_begin;
        #endif
        return;
    }

    void Speech_Engine::kws_asr_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        // check
        int ret = m_feature->check_data_length(data_len_samples);
        if (ret != 0)
        {
            printf("[ERROR:] %s, %d: Input data length is not 24000.\n", __FUNCTION__, __LINE__);
            return;
        }

        // get feature
        m_feature->get_featuer_slides_window(pdata, data_len_samples);

        for (int i = 0; i < m_feature->data_mat_time() - m_feature->feature_time(); i = i + m_feature->time_step_ms())
        {
            m_feature->get_single_feature(i);
            cv::Mat output_feature = m_feature->single_feature();

            std::cout << "\033[0;31m" << "[Information:] output_feature.rows: " << output_feature.rows << ", output_feature.cols: " << output_feature.cols <<"\033[0;39m" << std::endl;
            ASR::show_mat_uchar(output_feature, 120, 48);
        }
    }

} // namespace ASR