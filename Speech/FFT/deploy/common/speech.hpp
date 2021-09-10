#ifndef _ASR_SPEECH_H_
#define _ASR_SPEECH_H_

#include <memory>
#include <string>
#include <vector>

// #include "decode.hpp"
#include "feature.hpp"
// #include "model.hpp"

#ifdef _BUS
#include "webrtc/webrtc.hpp"
#endif

#ifdef _AIMESSAGE
#include "ai_message.hpp" 
#endif

namespace ASR
{
    struct ASR_PARAM_S
    {
        ASR_PARAM_S() : algorithem_id(0),
                        language_id(0),
                        streamanager_id(0),
                        n_fft(512),
                        nfilt(64),
                        sample_rate(16000),
                        receive_audio_length_s(1),
                        time_seg_ms(32),
                        time_step_ms(10),
                        feature_freq(64),
                        feature_time(96),
                        storage_audio_length_s(3),
                        storage_feature_time(296),
                        kws_weakup_model_name(""),
                        kws_weakup_param_name(""),
                        kws_weakup_input_name(""),
                        kws_weakup_output_name(""),
                        kws_weakup_feature_freq(64),
                        kws_weakup_feature_time(192),
                        kws_weakup_output_num(2),
                        kws_weakup_output_time(1),
                        kws_weakup_threshold(0.5),
                        kws_weakup_number_threshold(0.5),
                        kws_weakup_suppression_counter(3),
                        kws_weakup_asr_suppression_counter(1),
                        kws_model_name(""),
                        kws_param_name(""),
                        kws_input_name(""),
                        kws_output_name(""),
                        kws_feature_freq(64),
                        kws_feature_time(192),
                        kws_output_num(2),
                        kws_output_time(1),
                        asr_model_name(""),
                        asr_param_name(""),
                        asr_input_name(""),
                        asr_output_name(""),
                        asr_feature_freq(64),
                        asr_feature_time(296),
                        asr_output_num(408),
                        asr_output_time(40),
                        asr_suppression_counter(2),
                        asr_decode_id(0),
                        asr_dict_path(""),
                        asr_lm_path(""),
                        asr_lm_ngram_length(3),
                        asr_second_model_name(""),
                        asr_second_param_name(""),
                        asr_second_input_name(""),
                        asr_second_output_name(""),
                        asr_second_output_num(408),
                        asr_second_output_time(40),
                        asr_second_decode_id(0),
                        asr_second_dict_path(""),
                        asr_second_lm_path(""),
                        asr_second_lm_ngram_length(3), 
                        description(""){}

        int algorithem_id;      // 0：KWS; 1：ASR_16K; 2：ASR_8K; 3：Weakup_16K; 4：Weakup_ASR_16K; 5：Weakup_8K;
        int language_id;        // 0: chinese; 1: english; 
        int streamanager_id;        // 0: old; 1: new; 

        // nomal params
        int n_fft;
        int nfilt;
        int sample_rate;
        float receive_audio_length_s;
        int time_seg_ms;
        int time_step_ms;
        int feature_freq;
        int feature_time;
        int storage_audio_length_s;
        int storage_feature_time;

        // kws weakup model
        std::string kws_weakup_model_name;
        std::string kws_weakup_param_name;
        std::string kws_weakup_input_name;
        std::string kws_weakup_output_name;
        int kws_weakup_feature_freq;
        int kws_weakup_feature_time;
        int kws_weakup_output_num;
        int kws_weakup_output_time;
        float kws_weakup_threshold;
        float kws_weakup_number_threshold;
        int kws_weakup_suppression_counter;
        int kws_weakup_asr_suppression_counter;

        #ifdef _BUS
        int kws_weakup_activity_1st_counter=8;
        int kws_weakup_activity_2nd_counter=13;
        int kws_weakup_shield_counter=4;
        bool kws_weakup_remove_tts=false;
        std::string vad_mode="Aggressive";
        int vad_chunk_size=160;
        #endif

        // kws model
        std::string kws_model_name;
        std::string kws_param_name;
        std::string kws_input_name;
        std::string kws_output_name;
        int kws_feature_freq;
        int kws_feature_time;
        int kws_output_num;
        int kws_output_time;

        // asr model
        std::string asr_model_name;
        std::string asr_param_name;
        std::string asr_input_name;
        std::string asr_output_name;
        int asr_feature_freq;
        int asr_feature_time;
        int asr_output_num;
        int asr_output_time;
        int asr_suppression_counter;
        int asr_decode_id;

        std::string asr_dict_path;
        std::string asr_lm_path;
        int asr_lm_ngram_length;

        // special: asr second model
        std::string asr_second_model_name;
        std::string asr_second_param_name;
        std::string asr_second_input_name;
        std::string asr_second_output_name;
        int asr_second_output_num;
        int asr_second_output_time;
        int asr_second_decode_id;

        std::string asr_second_dict_path;
        std::string asr_second_lm_path;
        int asr_second_lm_ngram_length;

        std::string description;
    };

    struct Speech_Options_S
	{
        // container
        int data_container_ms = 100;              // 语音数据容器装有音频数据 100 ms
        int feature_data_ignore_samples = 10;     // 语音容器中语音数据对应语音特征的长度 100 ms -> 10，拼接特征时忽略的特征
        int feature_data_remove_samples = 6;      // 拼接特征需要丢弃的时间维度 6
        
        // kws
        int kws_forward_step_samples = 10;        // kws weakup 激活滑窗检测间隔

        // test 
        #ifdef _OUTPUTWAVE
        unsigned int output_recognized_wave_id = 0;           // 输出语音 id
        unsigned int output_weakup_wave_id = 0;           // 输出语音 id
        #endif
    };

    class Speech_Engine
    {
    public:
        // Speech_Engine();
        Speech_Engine(const ASR_PARAM_S &params, const char *file_path = NULL);
        ~Speech_Engine();

    public:
        void speech_recognition(short *pdata, int data_len_samples, char *output_keyword);
        
        #ifdef _BUS
        void activate_weakup_forcely();
        void deactivate_weakup_forcely();
        #endif

    protected:
        void init_feature_decode_model(const ASR_PARAM_S &params, const char *file_path = NULL);
        void normal_init(const ASR_PARAM_S &params);

        void normal_init_with_old_streamanager(const ASR_PARAM_S &params);
        void normal_init_with_new_streamanager(const ASR_PARAM_S &params);

        void asr_init(const ASR_PARAM_S &params, const char *file_path, std::string mAttr="AsrParam");
        void asr_second_init(const ASR_PARAM_S &params, const char *file_path, std::string mAttr="AsrPhonemeParam");
        void kws_weakup_init(const ASR_PARAM_S &params, const char *file_path);
        void kws_init(const ASR_PARAM_S &params, const char *file_path);
        void kws_asr_recognition(short *pdata, int data_len_samples, char *output_keyword);
        void asr_recognition(short *pdata, int data_len_samples, char *output_keyword);
        void kws_weakup_recognition(short *pdata, int data_len_samples, char *output_keyword);
        void kws_weakup_asr_recognition(short *pdata, int data_len_samples, char *output_keyword);

        #ifdef _BUS
        void kws_weakup_asr_recognition_for_bus(short *pdata, int data_len_samples, char *output_keyword);
        bool vad_process(short *pdata, int data_len_samples);
        #endif

        bool run_kws_weakup();
        void run_kws(std::vector<int> *result_id);
        void run_asr_after_kws(std::vector<int> *kws_id, std::string *output_str);
        void run_kws_asr(std::string &output_str);
        void run_asr(std::string &output_str, bool contorl_kws_bool=true);
        void run_asr_normal(std::string &output_str, bool contorl_kws_bool=true);
        void run_asr_second(std::string &output_str, bool contorl_kws_bool=true);
        int prepare_data_and_feature(short *pdata, int data_len_samples);

        void asr_duplicate_update_counter();
        std::string asr_duplicate_check(std::string output_str);

        void merge_data(short *pdata, int data_len_samples, short *audio_data);
        void store_data(short *pdata, int data_len_samples);
        void merge_feature_data(cv::Mat &output_feature);

        void merge_feature_data_with_old_streamanager(cv::Mat &output_feature);
        void merge_feature_data_with_new_streamanager(cv::Mat &output_feature);

        void merge_kws_score(const std::vector<float> &kws_scores, std::vector<float> *kws_merge_scores);
        void store_kws_score(const std::vector<float> &kws_scores);
        void init_kws_score_container();

        // test 
        #ifdef _OUTPUTWAVE
        void output_recognized_wave(std::string output_prefix_name="Weakup", std::string output_folder="/usr/local/audio_data/demo_test/");
        void output_weakup_wave(std::string output_prefix_name="AFTERWeakup", std::string output_folder="/usr/local/audio_data/demo_test/");
        #endif

    protected:
        // options
        Speech_Options_S m_speech_options;

        // feature
        std::unique_ptr<Feature> m_feature;

        // kws weakup model
        // std::unique_ptr<Decode> m_kws_weakup_decode;
        // std::unique_ptr<Model> m_kws_weakup_model;
        int m_kws_weakup_times;                     // weakup 模型滑窗次数，1s 语音音频，96 维语音特征，滑窗次数 = int(96 / kws_forward_step_samples) + 1
        int m_kws_weakup_suppression_counter_ms;       // kws weakup 激活抑制时间 1s
        int m_kws_weakup_asr_suppression_counter_ms;

        #ifdef _BUS
        int m_kws_weakup_activity_1st_counter_ms;
        int m_kws_weakup_activity_2nd_counter_ms;
        int m_kws_weakup_shield_counter_ms;
        bool m_kws_weakup_remove_tts;
        int m_kws_weakup_remove_tts_counter_ms = 0;
        bool m_kws_weakup_activate_forcely_flag;
        bool m_kws_weakup_deactivate_forcely_flag;

        std::unique_ptr<webrtc::Vad> m_voice_detector;
        int m_vad_chunk_size;
        int m_continous_silence_ms = 0;
        #endif

        // kws model 
        // std::unique_ptr<Decode> m_kws_decode;
        // std::unique_ptr<Model> m_kws_model;

        // asr model 
        // std::unique_ptr<Decode> m_asr_decode;
        // std::unique_ptr<Model> m_asr_model;
        int m_asr_decode_id;                        // 0: greedy; 1：beamsearch;
        // std::unique_ptr<Decode> m_asr_second_decode;
        // std::unique_ptr<Model> m_asr_second_model;
        int m_asr_second_decode_id;                 // 0: greedy; 1：beamsearch;
        int m_asr_suppression_counter_ms;              // asr 激活抑制时间 2s

        // container
        std::vector<short> m_data_container;        // 语音数据容器，用于拼接多秒语音特征
        cv::Mat m_feature_data_container;           // 语音特征容器
        std::vector<float> m_kws_score_container;   // kws 得分容器，用于滑窗输出结果
        int m_data_container_samples;               // 语音数据容器装有音频数据长度
        int m_kws_score_container_samples;          // kws 得分容器装有数据长度（注：根据语音唤醒模型滑窗次数决定 = m_kws_weakup_times） 

        // duplicate，防止重复检测
        std::map<std::string, int> m_asr_duplicate_counter_map;

        // on-off
        bool m_kws_weakup_bool;
        int m_kws_weakup_counter_ms;     //计数器，每次增加 m_receive_audio_length_ms, 累计到 m_kws_weakup_suppression_counter 后，做一次唤醒检测
        int m_weakup_asr_counter_ms;     //计数器，每次增加 m_receive_audio_length_ms, 累计到 m_kws_weakup_asr_suppression_counter 后，做一次ASR识别
        int m_asr_counter_ms;            //计数器，每次增加 m_receive_audio_length_ms, 累计到 m_asr_suppression_counter 后，做一次ASR识别

        int m_init_audio_length_ms;                  // 算法初始化阶段，语音时间长度
        int m_receive_audio_length_ms;               // 每次接收语音事件长度
        int m_storage_audio_length_ms;               // 算法初始化阶段，语音时间长度

        int m_algorithem_id;    // 0：KWS; 1：ASR_16K; 2：ASR_8K; 3：Weakup_16K; 4：Weakup_ASR_16K; 5：Weakup_8K;
        int m_language_id;      // 0: chinese; 1: english; 

        // 这几个参数是新的流管理方案中使用的参数
        int m_streamanager_id;
        int m_init_ignore_frames;
        int m_chunk_feature_frames;
        int m_feature_container_start_index=0;

        int m_sample_rate;                           // 采样率
        // test 
        #ifdef _OUTPUTWAVE
        // 方案一：只保存检出结果
        std::vector<short> m_output_recognized_data_container;  // 输出语音数据容器
        int m_output_recognized_data_container_samples;         // 语音数据容器装有音频数据长度

        // 方案二：保存唤醒后 9s 语音内容
        std::vector<short> m_output_weakup_data_container;      // 输出语音数据容器
        int m_output_weakup_audio_counter_ms;                   // 计数器，每次增加 m_receive_audio_length_ms, 累计到 m_output_weakup_audio_suppression_counter_ms 后，保存一次输出
        int m_output_weakup_audio_suppression_counter_ms = 6000; // 唤醒后等待指定时长后输出
        int m_output_weakup_audio_length_ms = 9000;             // 语音数据存储长度
        int m_output_weakup_data_container_samples;             // 语音数据容器装有音频数据长度
        #endif
    };

} // namespace ASR

#endif // _ASR_SPEECH_H_