#include <iostream>
#include <malloc.h>
#include <unistd.h>

#include "common.hpp"
#include "asr_config.h"
#include "speech.hpp"

#ifdef _TESTTIME
#ifndef TEST_TIME
#include <sys/time.h>
// #define TEST_TIME(times) do{ times = clock(); }while(0)
// #define TIME_DELTA(begin,end) 1000*(double((end)-(begin))/CLOCKS_PER_SEC)

#define TEST_TIME(times) do{\
        struct timeval cur_time;\
        gettimeofday(&cur_time, NULL);\
        times = (cur_time.tv_sec * 1000000llu + cur_time.tv_usec) / 1000llu;\
    }while(0)
#endif

#ifndef TIME_DELTA
#define TIME_DELTA(begin,end) (end)-(begin)
#endif

#ifdef _BUS
double vad_time = 0;
#endif 
double feature_time = 0;
double kws_weakup_model_forward_time = 0;
double kws_weakup_decode_time = 0;
double asr_normal_model_forward_time = 0;
double asr_normal_search_time = 0;    
double asr_normal_match_time = 0;    
double asr_second_model_forward_time = 0;
double asr_second_search_time = 0;
double asr_second_match_time = 0;
#endif

#ifdef _OUTPUTWAVE
#include <time.h>

#include "wave_data.hpp"
#endif

namespace ASR
{
    // Speech_Engine::Speech_Engine()
    // {
    //     m_feature.reset(new Feature);
    //     m_kws_weakup_decode.reset(new Decode);
    //     m_kws_weakup_model.reset(new Model);
    //     m_asr_decode.reset(new Decode);
    //     m_asr_model.reset(new Model);
    // }

    static Logger logger = getLogger();

    Speech_Engine::Speech_Engine(const ASR_PARAM_S &params, const char *file_path)
    {
        init_feature_decode_model(params, file_path);

        #ifdef _AIMESSAGE
        ai_message_init();
        #endif
    }

    Speech_Engine::~Speech_Engine()
    {
        m_feature.reset(nullptr);
        // m_asr_decode.reset(nullptr);
        // m_asr_model.reset(nullptr);
        // m_asr_second_decode.reset(nullptr);
        // m_asr_second_model.reset(nullptr);
        // m_kws_weakup_decode.reset(nullptr);
        // m_kws_weakup_model.reset(nullptr);
        // m_kws_decode.reset(nullptr);
        // m_kws_model.reset(nullptr);

        #ifdef _AIMESSAGE
        ai_message_uninit();
        #endif
    }
 
    void Speech_Engine::normal_init(const ASR_PARAM_S &params)
    {   
        m_streamanager_id = params.streamanager_id;
        if (params.streamanager_id == 0)
        {   
            //使用旧的流管理器时，目前的规则中，只限定每次接受一秒的数据
            logger.debug("Use old stream managing approach.");
            assert( params.receive_audio_length_s == 1.0 );
            normal_init_with_old_streamanager(params);
        }
        else if (params.streamanager_id == 1)
        {   
            logger.debug("Use new stream managing approach.");
            normal_init_with_new_streamanager(params);
        }
        else
        {
            logger.error("Unknow m_asr_streamanager_id.");
            return;
        }

        m_sample_rate = params.sample_rate;

        #ifdef _BUS
        if ( params.sample_rate != 8000 && params.sample_rate != 16000 &&  params.vad_chunk_size !=  160 && params.vad_chunk_size != 320 )
        {
            logger.error( "Voice Activity Detector only support sample rate = 8K|16K, chunk size = 160|320." );
        }
        else
        {
            if ( params.vad_mode == "Null" ) { m_voice_detector.reset(nullptr); }
            else
            {
                enum webrtc::Vad::Aggressiveness vad_mode;
                if (params.vad_mode == "Normal" ) { vad_mode = webrtc::Vad::kVadNormal; }
                else if (params.vad_mode == "LowBitrate") { vad_mode = webrtc::Vad::kVadLowBitrate; }
                else if (params.vad_mode == "Aggressive") { vad_mode = webrtc::Vad::kVadAggressive; }
                else if (params.vad_mode == "VeryAggressive") { vad_mode = webrtc::Vad::kVadVeryAggressive; }
                else 
                {
                    logger.error( "Unknown VAD mode!" );
                    m_voice_detector.reset(nullptr);
                    return;
                }

                m_voice_detector.reset(new webrtc::Vad( vad_mode ) );
                m_vad_chunk_size = params.vad_chunk_size;
                bool ret = m_voice_detector->Init();
                if ( ! ret ) 
                {
                    logger.error("Voice Activity Detector init failed.");
                    m_voice_detector.reset(nullptr);
                    return;
                }
                else
                {
                    logger.info( "Init voice detector sucessed!" );
                }
            }
        }
        #endif        
    }

    void Speech_Engine::normal_init_with_old_streamanager(const ASR_PARAM_S &params)
    {   
        // Params
        m_init_audio_length_ms = 0;
        m_receive_audio_length_ms = int(1000*params.receive_audio_length_s); 
        m_storage_audio_length_ms = int(1000*params.storage_audio_length_s); 

        m_algorithem_id = params.algorithem_id; 
        m_language_id = params.language_id; 

        m_data_container_samples = int(params.sample_rate * m_speech_options.data_container_ms / 1000.0);
        m_data_container.reserve(m_data_container_samples);
        for (int i = 0; i < m_data_container_samples; i++)
        {
            m_data_container.push_back(0);
        }
        m_feature_data_container = cv::Mat::zeros(params.storage_feature_time, params.feature_freq, CV_8UC1);

        // Feature
        logger.info("Init feature extractor");
        Feature_Options_S feature_options;
        feature_options.data_len_samples = (params.receive_audio_length_s + m_speech_options.data_container_ms /1000.0) * params.sample_rate;
        feature_options.feature_channels = 1;
        feature_options.n_fft = params.n_fft;
        feature_options.nfilt = params.nfilt;
        feature_options.sample_rate = params.sample_rate;
        feature_options.time_seg_ms = params.time_seg_ms;
        feature_options.time_step_ms = params.time_step_ms;
        feature_options.feature_freq = params.feature_freq;
        feature_options.feature_time = params.feature_time;
        feature_options.pcen_flag = false;
        feature_options.fft_flag =  (m_algorithem_id==0) ? true : false; 
        m_feature.reset(new Feature(feature_options));

        #ifdef _OUTPUTWAVE
        m_output_recognized_data_container_samples = int(params.sample_rate * m_storage_audio_length_ms / 1000);
        m_output_recognized_data_container.reserve(m_output_recognized_data_container_samples);

        m_output_weakup_audio_counter_ms = m_output_weakup_audio_suppression_counter_ms + m_receive_audio_length_ms; 
        m_output_weakup_data_container_samples = int(params.sample_rate * m_output_weakup_audio_length_ms / 1000);
        m_output_weakup_data_container.reserve(m_output_weakup_data_container_samples);
        #endif

        return;
    }

    void Speech_Engine::normal_init_with_new_streamanager(const ASR_PARAM_S &params)
    {
        m_init_audio_length_ms = 0;
        m_storage_audio_length_ms = int(1000*params.storage_audio_length_s); 
        m_algorithem_id = params.algorithem_id; 
        m_language_id = params.language_id; 
        m_receive_audio_length_ms = int(1000*params.receive_audio_length_s);

        // 容器总储存的数据长度必须可以被每次接收的数据长度整除
        assert( m_storage_audio_length_ms % m_receive_audio_length_ms == 0 );

        //
        int time_seg_samples = int( params.sample_rate*params.time_seg_ms/1000.0 );
        int time_step_samples = int( params.sample_rate*params.time_step_ms/1000.0 );

        // 自动计算特征容器长度
        // ( total points - window width ) / window shift 
        int storage_feature_frames = int((m_storage_audio_length_ms - params.time_seg_ms)/params.time_step_ms);

        // 计算需要缓存的流长度
        // 根据滑窗大小和距离，每次缓存的流数据的长度可能不等，
        // 但我们希望每次缓存的流数据是定长的，因此必须要保证，每次接受到的数据长度能被滑窗宽度整除
        assert( m_receive_audio_length_ms % params.time_step_ms == 0 );
        int receive_audio_length_samples = int(params.receive_audio_length_s * params.sample_rate);
        m_data_container_samples = receive_audio_length_samples - int((receive_audio_length_samples - time_seg_samples)/time_step_samples)*time_step_samples;
        m_data_container.reserve( m_data_container_samples );
        for (int i = 0; i < m_data_container_samples; i++)
        {
            m_data_container.push_back(0);
        }

        // 仅仅在第一次计算的时候，需要移除首部的几帧
        m_init_ignore_frames = int(m_data_container_samples/time_step_samples);
        m_chunk_feature_frames = (m_receive_audio_length_ms - params.time_seg_ms)/params.time_step_ms + m_init_ignore_frames;

        // 初始化特征缓存容器
        m_feature_data_container = cv::Mat::zeros(storage_feature_frames, params.feature_freq, CV_8UC1);

        Feature_Options_S feature_options;
        feature_options.data_len_samples = m_data_container_samples + receive_audio_length_samples;
        feature_options.feature_channels = 1;
        feature_options.n_fft = params.n_fft;
        feature_options.nfilt = params.nfilt;
        feature_options.sample_rate = params.sample_rate;
        feature_options.time_seg_ms = params.time_seg_ms;
        feature_options.time_step_ms = params.time_step_ms;
        feature_options.feature_freq = params.feature_freq;
        feature_options.pcen_flag = false;
        feature_options.fft_flag =  (m_algorithem_id==0) ? true : false; 

        m_feature.reset( new Feature(feature_options) );

        #ifdef _OUTPUTWAVE
        m_output_recognized_data_container_samples = int(params.sample_rate * m_storage_audio_length_ms / 1000);
        m_output_recognized_data_container.reserve(m_output_recognized_data_container_samples);
        
        m_output_weakup_audio_counter_ms = m_output_weakup_audio_suppression_counter_ms + m_receive_audio_length_ms; 
        m_output_weakup_data_container_samples = int(params.sample_rate * m_output_weakup_audio_length_ms / 1000);
        m_output_weakup_data_container.reserve(m_output_weakup_data_container_samples);
        #endif

        return;
    } 

    void Speech_Engine::asr_init(const ASR_PARAM_S &params, const char *file_path, std::string mAttr)
    {
        // check 
        if (params.storage_feature_time != params.asr_feature_time)
        {
            logger.error("Storage_feature_time do not equal asr_feature_time.");
            return;
        }

        // init
        rmCConfig cconfig;
        std::string s, tmp;
        std::ifstream infile;
        char config_file[256] = {0};

        // Params
        m_asr_suppression_counter_ms = int(1000*params.asr_suppression_counter);        
        m_asr_counter_ms = m_asr_suppression_counter_ms - m_receive_audio_length_ms;
        m_asr_decode_id = params.asr_decode_id;

        // Decode

        // Model

        return;
    }

    void Speech_Engine::asr_second_init(const ASR_PARAM_S &params, const char *file_path, std::string mAttr)
    {
        // init
        rmCConfig cconfig;
        std::string s, tmp;
        std::ifstream infile;
        char config_file[256] = {0};

        // Params
        m_asr_second_decode_id = params.asr_second_decode_id;

        // Decode

        // Model
        
        return;
    }

    void Speech_Engine::kws_weakup_init(const ASR_PARAM_S &params, const char *file_path)
    {
        m_kws_weakup_bool = false;
        // Params
        m_kws_weakup_times = int(float(params.feature_time) / float(m_speech_options.kws_forward_step_samples)) + 1;
        m_kws_weakup_suppression_counter_ms = int(1000*params.kws_weakup_suppression_counter);
        m_kws_weakup_asr_suppression_counter_ms = int(1000*params.kws_weakup_asr_suppression_counter);
        m_kws_score_container_samples = m_kws_weakup_times;
        m_kws_weakup_counter_ms = m_kws_weakup_suppression_counter_ms - m_receive_audio_length_ms;
        m_weakup_asr_counter_ms = 0;

        m_kws_score_container.reserve(m_kws_score_container_samples);
        init_kws_score_container();

        // Decode  
    
        #ifdef _BUS
        m_kws_weakup_activity_1st_counter_ms = 1000*params.kws_weakup_activity_1st_counter;
        m_kws_weakup_activity_2nd_counter_ms = 1000*params.kws_weakup_activity_2nd_counter;
        m_kws_weakup_shield_counter_ms = 1000*params.kws_weakup_shield_counter;
        m_kws_weakup_remove_tts = params.kws_weakup_remove_tts;
        m_kws_weakup_activate_forcely_flag = false;
        m_kws_weakup_deactivate_forcely_flag = false;        
        #endif
    }

    void Speech_Engine::kws_init(const ASR_PARAM_S &params, const char *file_path)
    {
        // Decode

        // Model

    }

    void Speech_Engine::init_feature_decode_model(const ASR_PARAM_S &params, const char *file_path)
    {        
        logger.info( "Init Feature && Decode && Model Start." );

        // check 
        if (file_path == nullptr)
        {
            logger.error("Load param, miss resource file path.");
            return;
        }
        if (params.algorithem_id == 0)
        {
            normal_init(params);
            asr_init(params, file_path);
            kws_init(params, file_path);
        }
        else if ((params.algorithem_id == 1) || (params.algorithem_id == 2)) 
        {
            normal_init(params);
            asr_init(params, file_path);
        }
        else if (params.algorithem_id == 4)
        {   
            normal_init(params);
            asr_init(params, file_path);
            kws_weakup_init(params, file_path);
        }
        else if ((params.algorithem_id == 3) || (params.algorithem_id == 5))
        {   
            normal_init(params);
            kws_weakup_init(params, file_path);
        }
        else if (params.algorithem_id == 6)
        {   
            normal_init(params);
            asr_init(params, file_path, "AsrParam");
            asr_second_init(params, file_path, "AsrSecondParam");
            kws_weakup_init(params, file_path);
        }
        else {
            logger.error("Unknown algorithem_id.");
            return;
        }

        logger.info( "Init Feature && Decode && Model Done." );
    }

    void Speech_Engine::speech_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        //std::cout<<"====RMAPI_AI_ASR_RUN_Start====="<<std::endl;
        if (m_algorithem_id == 0)
        {
            kws_asr_recognition(pdata, data_len_samples, output_keyword);
        }
        else if ((m_algorithem_id == 1) || (m_algorithem_id == 2)) 
        {
            asr_recognition(pdata, data_len_samples, output_keyword);
        }
        else if ((m_algorithem_id == 4) || (m_algorithem_id == 6))
        {
            #ifdef _BUS
            // 如果需要移除TTS反应时间，直接在此处丢弃，不参与后续计算
            if ( m_kws_weakup_remove_tts && m_kws_weakup_remove_tts_counter_ms > 0 )
            {
                m_kws_weakup_remove_tts_counter_ms -= m_receive_audio_length_ms;
                        
                // output
                sprintf(output_keyword, "%s", "<TTS> ");

                logger.debug( "** Discard TTS voice." );

                return;
            }

            kws_weakup_asr_recognition_for_bus(pdata, data_len_samples, output_keyword);
            #else
            kws_weakup_asr_recognition(pdata, data_len_samples, output_keyword);
            #endif
        }
        else if ((m_algorithem_id == 3) || (m_algorithem_id == 5))
         {
            kws_weakup_recognition(pdata, data_len_samples, output_keyword);
        }
        else {
            logger.error("Unknown algorithem_id.");
            return;
        }

    }

    void Speech_Engine::merge_data(short *pdata, int data_len_samples, short *audio_data)
    {
        for (unsigned int i = 0; i < m_data_container.size(); i++)
        {
            audio_data[i] = m_data_container[i];
        }
        for (int i = 0; i < data_len_samples; i++)
        {
            audio_data[i + m_data_container.size()] = pdata[i];
        }
    }

    void Speech_Engine::store_data(short *pdata, int data_len_samples)
    {
        m_data_container.clear();
        for (int i = m_data_container_samples - 1; i >= 0; i--)
        {
            m_data_container.push_back(pdata[data_len_samples - 1 - i]);
        }

        #ifdef _OUTPUTWAVE
        // 储存音频，用于保存音频
        if (m_output_recognized_data_container.size() >= static_cast<unsigned int>(m_output_recognized_data_container_samples))
        {
            for (int i = 0; i < m_output_recognized_data_container_samples - data_len_samples; i++)
            {
                m_output_recognized_data_container[i] = m_output_recognized_data_container[i + data_len_samples];
            }
            for (int i = 0; i < data_len_samples; i++)
            {
                m_output_recognized_data_container[m_output_recognized_data_container_samples - data_len_samples + i] = pdata[i];
            }
        }
        else
        {
            for (int i = 0; i < data_len_samples; i++)
            {
                m_output_recognized_data_container.push_back(pdata[i]);
            }
        }
        if (m_output_weakup_data_container.size() >= static_cast<unsigned int>(m_output_weakup_data_container_samples))
        {
            for (int i = 0; i < m_output_weakup_data_container_samples - data_len_samples; i++)
            {
                m_output_weakup_data_container[i] = m_output_weakup_data_container[i + data_len_samples];
            }
            for (int i = 0; i < data_len_samples; i++)
            {
                m_output_weakup_data_container[m_output_weakup_data_container_samples - data_len_samples + i] = pdata[i];
            }
        }
        else
        {
            for (int i = 0; i < data_len_samples; i++)
            {
                m_output_weakup_data_container.push_back(pdata[i]);
            }
        }
        #endif
        return;
    }

    void Speech_Engine::merge_feature_data(cv::Mat &output_feature)
    {
        if (m_streamanager_id == 0)
        {   
            merge_feature_data_with_old_streamanager(output_feature);
        }
        else if (m_streamanager_id == 1)
        {
            merge_feature_data_with_new_streamanager(output_feature);
        }
        else
        {
            logger.error("Unknown asr_streamanager_id.");
            return;
        }
    }

    void Speech_Engine::merge_feature_data_with_old_streamanager(cv::Mat &output_feature)
    {   
        if (m_init_audio_length_ms == 0)
        {
            cv::Mat feature_data_rows  = m_feature_data_container.rowRange(0, output_feature.rows - m_speech_options.feature_data_ignore_samples);
            cv::Mat output_feature_rows  = output_feature.rowRange(m_speech_options.feature_data_ignore_samples, output_feature.rows);
            output_feature_rows.copyTo(feature_data_rows);
            m_init_audio_length_ms += 1000;
        }
        else if (m_init_audio_length_ms < m_storage_audio_length_ms)
        {
            cv::Mat feature_data_rows  = m_feature_data_container.rowRange(((m_init_audio_length_ms/1000.0) * (output_feature.rows - m_speech_options.feature_data_remove_samples)) - m_speech_options.feature_data_ignore_samples, \
                                         (((m_init_audio_length_ms/1000.0) + 1) * output_feature.rows) - m_speech_options.feature_data_ignore_samples - ((m_init_audio_length_ms/1000.0) * m_speech_options.feature_data_remove_samples));
            output_feature.copyTo(feature_data_rows);
            m_init_audio_length_ms += 1000;
        }
        else if (m_init_audio_length_ms == m_storage_audio_length_ms)
        {
            cv::Mat feature_data_remain_rows  = m_feature_data_container.rowRange(output_feature.rows - m_speech_options.feature_data_remove_samples, m_feature_data_container.rows).clone();
            cv::Mat feature_data_rows  = m_feature_data_container.rowRange(0, (((m_init_audio_length_ms/1000.0) - 1) * output_feature.rows) - m_speech_options.feature_data_ignore_samples - (((m_init_audio_length_ms/1000.0) - 2) * m_speech_options.feature_data_remove_samples));
            feature_data_remain_rows.copyTo(feature_data_rows);
            feature_data_rows  = m_feature_data_container.rowRange((((m_init_audio_length_ms/1000.0) - 1) * (output_feature.rows - m_speech_options.feature_data_remove_samples)) - m_speech_options.feature_data_ignore_samples, m_feature_data_container.rows);
            output_feature.copyTo(feature_data_rows);
        }
        return;
    }

    void Speech_Engine::merge_feature_data_with_new_streamanager(cv::Mat &output_feature)
    {   
        // 新的到的特征矩阵的frames数必须等于理论计算到的帧数
        assert( output_feature.rows == m_chunk_feature_frames );

        if (m_init_audio_length_ms == 0)
        {
           // 如果这是第0次计算
           cv::Mat feature_data_rows = m_feature_data_container.rowRange(0, m_chunk_feature_frames - m_init_ignore_frames );
           cv::Mat output_feature_rows = output_feature.rowRange(m_init_ignore_frames, m_chunk_feature_frames);
           output_feature_rows.copyTo(feature_data_rows);
           m_init_audio_length_ms += m_receive_audio_length_ms;
           m_feature_container_start_index +=  (m_chunk_feature_frames - m_init_ignore_frames);
           //std::cout << "here 11:" << m_init_audio_length_s << "," << m_receive_audio_length_s << std::endl;
        }   
        else if (m_init_audio_length_ms < m_storage_audio_length_ms)
        {  
           // 如果数据还没存满容器，则直接在后面追加
           cv::Mat feature_data_rows  = m_feature_data_container.rowRange(m_feature_container_start_index, m_feature_container_start_index+m_chunk_feature_frames);
           output_feature.copyTo(feature_data_rows);
           m_init_audio_length_ms += m_receive_audio_length_ms;
           m_feature_container_start_index += m_chunk_feature_frames;
            
        //    if (m_init_audio_length_s == m_storage_audio_length_s)
        //    {
        //         // 保存数据
        //         cv::FileStorage fs("/home/engineers/wy_rmai/wy_code/demo/compare_feature/test_feature_cpp_new.xml", cv::FileStorage::WRITE);
        //         fs << "feature" << m_feature_data_container;
        //         fs.release(); 
        //         std::cout << "Save chunk feature done:" << m_init_audio_length_s << std::endl;
        //    }
        }
        else if (m_init_audio_length_ms == m_storage_audio_length_ms)
        {   
            //如果特征已经存满了容器，则先将cover的部分前移，
            int cover_size = m_feature_data_container.rows - m_chunk_feature_frames;
            cv::Mat feature_data_remain_rows  = m_feature_data_container.rowRange( m_chunk_feature_frames, m_feature_data_container.rows ).clone();
            cv::Mat feature_data_rows  = m_feature_data_container.rowRange( 0, cover_size  );
            feature_data_remain_rows.copyTo(feature_data_rows);
            feature_data_rows  = m_feature_data_container.rowRange(cover_size, m_feature_data_container.rows);
            output_feature.copyTo(feature_data_rows);       
            // if (m_init_audio_length_s == m_storage_audio_length_s)
            // {
            //     // 保存数据
            //     cv::FileStorage fs("/home/engineers/wy_rmai/wy_code/demo/compare_feature/test_feature_cpp_new.xml", cv::FileStorage::WRITE);
            //     fs << "feature" << m_feature_data_container;
            //     fs.release(); 
            //     std::cout << "Save chunk feature done:" << m_init_audio_length_s << std::endl;
            // }
            // throw "wangyu";     
        }
        else
        {
            logger.error("Invalid storage audio length counter.");
            return;
        }

    }        

    void Speech_Engine::merge_kws_score(const std::vector<float> &kws_scores, std::vector<float> *kws_merge_scores)
    {
        (*kws_merge_scores).clear();
        for (int i = 0; i < m_kws_score_container_samples; i++)
        {
            (*kws_merge_scores).push_back(m_kws_score_container[i]);
        }
        for (int i = 0; i < m_kws_score_container_samples; i++)
        {
            (*kws_merge_scores).push_back(kws_scores[i]);
        }
        m_kws_score_container.clear();
    }

    void Speech_Engine::store_kws_score(const std::vector<float> &kws_scores)
    {
        m_kws_score_container.clear();
        for (int i = 0; i < m_kws_score_container_samples; i++)
        {
            m_kws_score_container.push_back(kws_scores[i]);
        }
    }

    void Speech_Engine::init_kws_score_container()
    {
        m_kws_score_container.clear();
        for (int i = 0; i < m_kws_score_container_samples; i++)
        {
            m_kws_score_container.push_back(0.0);
        }
    }

    int Speech_Engine::prepare_data_and_feature(short *pdata, int data_len_samples)
    {   
        #ifdef _BUS
        if ( m_voice_detector != nullptr )
        {   
            #ifdef _TESTTIME
            unsigned long long time_begin_vad, time_end_vad;
            TEST_TIME(time_begin_vad);
            #endif
            bool is_silence = vad_process(pdata, data_len_samples);
            if ( is_silence )
            {
                if ( m_continous_silence_ms == m_storage_audio_length_ms - m_receive_audio_length_ms )
                {
                    //如果已经连续是静音，将不参与特征计算和模型前传
                    logger.debug( "Continuous silences detected. Skip feature extraction and model forward!" );
                    return 1;
                }
                else
                {
                    m_continous_silence_ms += m_receive_audio_length_ms;
                    logger.debug( "Silence detected!" );
                }
            }
            else
            {
                m_continous_silence_ms = 0;
            }
            #ifdef _TESTTIME
            TEST_TIME(time_end_vad);
            vad_time += TIME_DELTA(time_begin_vad,time_end_vad);
            #endif

            // [TODO] res == true， 代表整段语音就是silence
        }
        #endif   

        int audio_data_length = data_len_samples + m_data_container_samples; // 16000 + 1600
        short audio_data[audio_data_length]; 
        for (int i=0; i<audio_data_length; i++ )
        {
            audio_data[i] = 0;
        }

        int ret = m_feature->check_data_length(audio_data_length);
        if (ret != 0)
        {
            logger.error("Input data length is not right.");
            return -1;
        }

        // audio data
        // 拼接语音数据
        merge_data(pdata, data_len_samples, audio_data);
        // std::cout << "\033[0;31m" << "[Information:] audio_data: " << "\033[0;39m" << std::endl;
        // for(unsigned int i = 0; i < 2000; i ++) {
        //     std::cout << audio_data[i]<< " ";
        // }
        // std::cout << std::endl;

        // 存储指定时间的音频，用于后续拼接语音特征
        store_data(pdata, data_len_samples);
        // std::cout << "\033[0;31m" << "[Information:] m_data_container: " << "\033[0;39m" << std::endl;
        // for(unsigned int i = 0; i < m_data_container_samples; i ++) {
        //     std::cout << m_data_container[i]<< " ";
        // }
        // std::cout << std::endl;

        #ifdef _TESTTIME
        unsigned long long time_begin, time_end;
        TEST_TIME(time_begin);
        #endif

        // feature
        // 计算特征
        m_feature->get_featuer_total_window(audio_data, audio_data_length);
        cv::Mat output_feature = m_feature->mfsc_feature_int();
        // std::cout << "[Information:] output_feature.rows: " << output_feature.rows << ", output_feature.cols: " << output_feature.cols << std::endl;
        // ASR::show_mat_uchar(output_feature, 106, 64);

        // 拼接特征
        merge_feature_data(output_feature);
        // std::cout << "[Information:] m_feature_data_container.rows: " << m_feature_data_container.rows << ", m_feature_data_container.cols: " << m_feature_data_container.cols << std::endl;
        // ASR::show_mat_uchar(m_feature_data_container, 296, 64);

        #ifdef _TESTTIME
        TEST_TIME(time_end);
        feature_time += TIME_DELTA(time_begin,time_end);
        #endif

        return 0;
    }

    void Speech_Engine::kws_asr_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        // init 
        std::string output_str;

        // 准备数据和特征
        int ret = prepare_data_and_feature(pdata, data_len_samples);
        if (ret != 0)
        {
            logger.error("Input data length is not right.");
            return;
        }

        // 如果语音特征未装满容器，不进行唤醒和关键词检测
        if (m_init_audio_length_ms < m_storage_audio_length_ms)
            return;

        // // asr_duplicate_update_counter，更新计数器，防止重复检测
        // asr_duplicate_update_counter();

        // 间隔 m_asr_suppression_counter 进行一次检测
        m_asr_counter_ms += m_receive_audio_length_ms;
        if (m_asr_counter_ms < m_asr_suppression_counter_ms)
        {
            return;
        }
        m_asr_counter_ms = 0;

        // forward
        run_kws_asr(output_str);
        if (output_str.length() > 0)
        {
            logger.debug( "===============!!!!!!!!!!!!!!===============" );
            logger.debug( "********************************************" );
            logger.debug( "**" );
            logger.info( std::string("** Detect Command: \033[0;31m ") + output_str + "\033[0;39m"  );
            logger.debug( "**" );
            logger.debug( "********************************************" );
        
            #ifdef _OUTPUTWAVE
            output_recognized_wave("Asr_" + output_str);
            #endif
        }
        else
        {
            logger.debug( "** [Information:] Detecting ... " );
        }

        // output
        sprintf(output_keyword, "%s", output_str.c_str());
        return;
    }

    void Speech_Engine::asr_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        // init 
        std::string output_str;

        // 准备数据和特征
        int ret = prepare_data_and_feature(pdata, data_len_samples);
        if (ret != 0)
        {
            return;
        }

        // 如果语音特征未装满容器，不进行唤醒和关键词检测
        if (m_init_audio_length_ms < m_storage_audio_length_ms)
            return;

        // asr_duplicate_update_counter，更新计数器，防止重复检测
        asr_duplicate_update_counter();

        // 间隔 m_asr_suppression_counter 进行一次检测
        m_asr_counter_ms += m_receive_audio_length_ms;
        if (m_asr_counter_ms < m_asr_suppression_counter_ms)
        {
            return;
        }
        m_asr_counter_ms = 0;

        // forward
        run_asr(output_str, false);
        if (output_str.length() > 0)
        {
            logger.debug( "===============!!!!!!!!!!!!!!===============" );
            logger.debug( "********************************************" );
            logger.debug( "**" );
            logger.info( std::string("** Detect Command: \033[0;31m ") + output_str + "\033[0;39m"  );
            logger.debug( "**" );
            logger.debug( "********************************************" );
        
            #ifdef _OUTPUTWAVE
            output_recognized_wave("Asr_" + output_str);
            #endif
        }
        else
        {
            logger.debug( "** Detecting ... " );
        }

        // output
        sprintf(output_keyword, "%s", output_str.c_str());
        return;
    }

    void Speech_Engine::kws_weakup_asr_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        // init 
        std::string output_str;
        
        // 准备数据和特征
        int ret = prepare_data_and_feature(pdata, data_len_samples);
        if (ret != 0)
        {
            return;
        }

        // 如果语音特征未装满容器，不进行唤醒和关键词检测
        if (m_init_audio_length_ms < m_storage_audio_length_ms)
            return;

        // asr_duplicate_update_counter，更新计数器，防止重复检测
        asr_duplicate_update_counter();

        // 方案一：进行 kws weakup 唤醒词检测，若检测到唤醒词，未来三秒进行 asr 检测
        if (not m_kws_weakup_bool)
        {   
            m_kws_weakup_counter_ms += m_receive_audio_length_ms;
            if ( m_kws_weakup_counter_ms >= m_kws_weakup_suppression_counter_ms )
            {
                m_kws_weakup_counter_ms = 0;

                bool find_kws_weakup_bool = run_kws_weakup();

                if (find_kws_weakup_bool)
                {
                    m_kws_weakup_bool = true;

                    logger.debug( "===============!!!!!!!!!!!!!!===============" );
                    logger.debug( "********************************************" );
                    logger.debug( "**" );
                    logger.info( "** Device Weakup: \033[0;31m Weakup \033[0;39m" );
                    logger.debug( "**" );
                    logger.debug( "********************************************" );
                    output_str += "Weakup ";

                    #ifdef _AIMESSAGE
                    std::string message_string;
                    Json::Value message_json = create_message_json();
                    json_to_string(message_json, message_string);
                    ret = send_msg((void *)message_string.c_str(), (int)strlen(message_string.c_str()));\
                    if (ret != 0)
                    {
                        logger.error("Send_message is not right.");
                    }
                    // std::cout << "** [Information:] message_string: \033[0;31m" << message_string << "\033[0;39m" << std::endl;
                    #endif

                    #ifdef _OUTPUTWAVE
                    output_recognized_wave();

                    if ( m_output_weakup_audio_counter_ms >= m_output_weakup_audio_suppression_counter_ms )
                        m_output_weakup_audio_counter_ms = 0;
                    #endif
                }
            }
        }
        else
        {
            m_weakup_asr_counter_ms += m_receive_audio_length_ms;
            if (m_weakup_asr_counter_ms >= m_kws_weakup_asr_suppression_counter_ms)
            {
                m_weakup_asr_counter_ms = 0;
                m_asr_counter_ms -= m_receive_audio_length_ms;
                m_kws_weakup_bool = false;

                run_asr(output_str, true);
                if (output_str.length() > 0)
                {
                    logger.debug( "===============!!!!!!!!!!!!!!===============" );
                    logger.debug( "********************************************" );
                    logger.debug( "**" );
                    logger.info( std::string("** Detect Command: \033[0;31m ") + output_str + "\033[0;39m"  );
                    logger.debug( "**" );
                    logger.debug( "********************************************" );

                    #ifdef _OUTPUTWAVE
                    output_recognized_wave("Asr_" + output_str);
                    #endif
                }
                else
                {
                    logger.debug( "** Detecting ... " );
                }
            }
        }

        // 方案二：进行 asr 检测，间隔一定时长进行检测
        if (m_kws_weakup_bool)
            m_asr_counter_ms = 0;
        else 
            m_asr_counter_ms += m_receive_audio_length_ms;

        if (m_asr_counter_ms >= m_asr_suppression_counter_ms)
        {
            m_asr_counter_ms = 0;
            run_asr(output_str, false);
            if (output_str.length() > 0)
            {
                logger.debug( "===============!!!!!!!!!!!!!!===============" );
                logger.debug( "********************************************" );
                logger.debug( "**" );
                logger.info( std::string("** Detect Command: \033[0;31m ") + output_str + "\033[0;39m"  );
                logger.debug( "**" );
                logger.debug( "********************************************" );

                #ifdef _OUTPUTWAVE
                output_recognized_wave("Asr_" + output_str);
                #endif
            }
            else
            {
                logger.debug( "** Detecting ... " );
            }
        }

        // output
        sprintf(output_keyword, "%s", output_str.c_str());

        #ifdef _OUTPUTWAVE
        if ( m_output_weakup_audio_counter_ms <= m_output_weakup_audio_suppression_counter_ms )
        {
            m_output_weakup_audio_counter_ms += m_receive_audio_length_ms;

            if ( m_output_weakup_audio_counter_ms > m_output_weakup_audio_suppression_counter_ms ) 
                output_weakup_wave();
        }
        #endif
    }
    
    #ifdef _BUS
    void Speech_Engine::activate_weakup_forcely()
    {   
        m_kws_weakup_activate_forcely_flag = true;
    }

    void Speech_Engine::deactivate_weakup_forcely()
    {
        if ( m_kws_weakup_bool )
        {
            m_kws_weakup_deactivate_forcely_flag = true;
        }
        else
        {
            m_kws_weakup_deactivate_forcely_flag = false;
        }
    }

    void Speech_Engine::kws_weakup_asr_recognition_for_bus(short *pdata, int data_len_samples, char *output_keyword)
    {   
        // 2021 06 09, wang yu
        // 这是一个为了适配公交车智能语音交互一体机第二版需求而设计的唤醒加识别的新逻辑

        // init 
        std::string output_str;
        
        // #####################################
        // # 准备数据和特征
        // #####################################

        int ret_code = prepare_data_and_feature(pdata, data_len_samples);
        // ret_code == 0: 数据准备完成，正常进行检测和识别
        // ret_code == 1: 被判定为静音，不运行检测和识别模型，但要进行其他的逻辑步骤
        // ret_code < 0: 准备数据和特征阶段发生了error
        if ( ret_code < 0 ) 
        { 
            return; 
        }
        // 如果语音特征未装满容器，不进行唤醒和关键词检测
        if (m_init_audio_length_ms < m_storage_audio_length_ms)
            return;
        
        // ###########################################################################################
        // # 方案一: 唤醒 + 识别
        // ###########################################################################################

        // ----------------------------------------KWS Weakup 步骤----------------------------------------------

        // 1. 如果当前不是唤醒状态，看是否被主动唤醒或则做一次唤醒检测
        // 2. 如果当前是唤醒状态，但已经超过了屏蔽时间，仍然可以被再次唤醒，所以做唤醒检测
        // 3. 如果当前是唤醒状态，被再次主动唤醒的情况，此时不做唤醒检测，但需要做初始化
        if ( ( m_kws_weakup_bool == false ) || 
             ( m_weakup_asr_counter_ms > m_kws_weakup_shield_counter_ms ) ||
             ( m_kws_weakup_activate_forcely_flag == true ) )
        {   
            // 看一次计数器是否超过抑制时长 或者 是否被主动唤醒
            m_kws_weakup_counter_ms += m_receive_audio_length_ms;
            if ( (m_kws_weakup_counter_ms >= m_kws_weakup_suppression_counter_ms) || (m_kws_weakup_activate_forcely_flag == true))
            {
                m_kws_weakup_counter_ms = 0;
                // 如果没有被被主动唤醒（case 1和2），尝试做一次唤醒检测
                bool find_kws_weakup_bool = m_kws_weakup_activate_forcely_flag;
                if ( m_kws_weakup_activate_forcely_flag == false )
                {   
                    find_kws_weakup_bool = ( ret_code == 1 ? false : run_kws_weakup() ); 
                }

                // 如果成功进入唤醒状态(主动或者被检测到)，做一些初始化的操作
                // 1. 重置计数器
                // 2. 清空缓存特征，和解码器缓存的未被匹配到的拼音
                if (find_kws_weakup_bool)
                {
                    m_kws_weakup_bool = true;
                    m_weakup_asr_counter_ms = 0;
                    m_feature_data_container.setTo( cv::Scalar(0) );
                    m_asr_decode->clear_chinese_result_cache();
                    // 接下来要移除一秒的TTS
                    m_kws_weakup_remove_tts_counter_ms = 1000;

                    logger.debug( "===============!!!!!!!!!!!!!!===============" );
                    logger.debug( "********************************************" );
                    logger.debug( "**" );
                    logger.info( "** Device Weakup: \033[0;31m <WEAKUP> \033[0;39m"  );
                    logger.debug( "**" );
                    logger.debug( "********************************************" );
                    output_str += "<WEAKUP> ";

                    #ifdef _OUTPUTWAVE
                    output_recognized_wave();
                    #endif

                }
                // 如果没检测到新的唤醒，但现在本身就是唤醒状态（ case 2,3 ）
                // 1. 累计一次计数，并跳转去【唤醒后的ASR步骤】
                else if (m_weakup_asr_counter_ms > m_kws_weakup_shield_counter_ms)
                {
                    m_weakup_asr_counter_ms += m_receive_audio_length_ms ;
                }
                
                // 重置主动激活的flag
                m_kws_weakup_activate_forcely_flag = false;
            }
        }
        // 如果当前已经进入唤醒状态，并且不需要进行kws检测（即未超过屏蔽时间）
        // 1. 累计一次计数，并跳转去【唤醒后的ASR步骤】
        else
        {
            m_weakup_asr_counter_ms += m_receive_audio_length_ms;
        }

        // ----------------------------------------唤醒后的 ASR 步骤----------------------------------------------
        if (m_kws_weakup_bool == true)
        {   
            // 如果唤醒状态被主动退出
            // 1. 重置计数器
            // 2. 清空特征和解码器缓存
            if  ( m_kws_weakup_deactivate_forcely_flag == true )
            {   
                // 重置flag
                m_kws_weakup_deactivate_forcely_flag = false;
                // 输出一个结束标记
                output_str.append( "<OVER>" );
                // 重置计数器和一些其他的flag
                m_kws_weakup_counter_ms = 0;
                m_weakup_asr_counter_ms = 0;
                m_kws_weakup_bool = false;
                m_kws_weakup_remove_tts_counter_ms = 0;
                m_feature_data_container.setTo( cv::Scalar(0) );
                m_asr_decode->clear_chinese_result_cache();
            }
            // 否则，如果采集到新的数据
            else if (m_weakup_asr_counter_ms > 0 )
            {
                // 如果满足时间间隔，做ASR
                if ( m_weakup_asr_counter_ms % m_kws_weakup_asr_suppression_counter_ms == 0 )
                {   
                    if ( ret_code == 1 ) 
                    { 
                        //strcpy(output_str, "<SIL>");
                        output_str = "<SIL>";
                    }
                    else 
                    { 
                        run_asr(output_str, true); 
                    }
                    // 再此版本中 output_str 一定有值，并为下面四种情况之一:
                    // 1. "<SIL>"，表示静音
                    // 2. "<UNK>", 表示这个词在词典之外，是未知词 
                    // 3. "control_开始考勤"，表示这是一个控制词
                    // 4. "要加钱", 表示这个词在词典之内，但不是一个控制词
                    
                    logger.debug( "===============!!!!!!!!!!!!!!===============" );
                    logger.debug( "********************************************" );
                    logger.debug( "**" );
                    logger.info( std::string("** Detect Command: \033[0;31m ") + output_str + "\033[0;39m"  );
                    logger.debug( "**" );
                    logger.debug( "********************************************" );

                    //如果出现了控制词
                    if ( output_str.find("control_") != std::string::npos )
                    {
                        // 重置计数，退出激活状态
                        m_weakup_asr_counter_ms = 0;
                        output_str.append( "<OVER>" );
                        m_kws_weakup_bool = false;
                        // 清空缓存
                        m_feature_data_container.setTo( cv::Scalar(0) );
                        // 重置 TTS flag
                        m_kws_weakup_remove_tts_counter_ms = 0;

                        #ifdef _OUTPUTWAVE
                        output_recognized_wave("Asr_" + output_str);
                        #endif
                    }
                    //如果出现<UNK>或者不是控制词
                    else if ( output_str.find("<SIL>") == std::string::npos )
                    {
                        m_weakup_asr_counter_ms = 0;
                        // 清空缓存
                        m_feature_data_container.setTo( cv::Scalar(0) );
                    }

                    // 如果已经超过了警告时间，则警告
                    // 必须要验证是否能被整除
                    if ( m_weakup_asr_counter_ms == m_kws_weakup_activity_1st_counter_ms )
                    {   
                        // 输出一个警告标记
                        output_str.append("<WARNING> ");
                        // 特征缓存不需要特意清空
                        m_feature_data_container.setTo( cv::Scalar(0) );
                        // 解码器缓存已经被解码器自己清空了
                        // 接下来要移除2秒的TTS
                        m_kws_weakup_remove_tts_counter_ms = 2000;
                    }
                    // 如果已经超过了结束时间，则强制退出
                    else if ( m_weakup_asr_counter_ms == m_kws_weakup_activity_2nd_counter_ms )
                    {
                        // 输出一个结束标记
                        output_str.append("<OVER> ");
                        // 重置计数器
                        m_weakup_asr_counter_ms = 0;
                        m_kws_weakup_bool = false;
                        // 重设TTS flag 
                        m_kws_weakup_remove_tts_counter_ms = 0;
                        // 特征缓存不需要特意清空
                        m_feature_data_container.setTo( cv::Scalar(0) );
                        // 解码器缓存已经被解码器自己清空了
                    }

                }                
            }
        }

        // bus 不运行第二套方案
        // output
        sprintf(output_keyword, "%s", output_str.c_str());
    }

    bool Speech_Engine::vad_process(short *pdata, int data_len_samples)
    {
        // 做VAD操作，如果这一段语音被检测为静音
        // 强制设置值为 0
        int16_t buffer[m_vad_chunk_size];
        int N = data_len_samples/m_vad_chunk_size;
        
        if ( data_len_samples != N * m_vad_chunk_size ) 
        {   
            logger.debug( "Data length is not integral times of vad chunk size. Skip VAD!" );
            return false;
        }
        
        int silence_counter = 0;
        //int debug_u_counter = 0;

        // 一段一段处理
        for ( int i=0; i<N; i++ )
        {   
            // 拷贝数据
            for ( int j=0; j<m_vad_chunk_size; j++ )
            {
                buffer[ j ] =  pdata[ i*m_vad_chunk_size + j ];
            }
            // 做vad
            int res = m_voice_detector->IsSpeech( buffer, m_vad_chunk_size, m_sample_rate );
            if ( res < 0 ) {
                logger.error("VAD process failed. Set it to: activaty!");
                res = 1;
            }
            // 如果是静音，则将元数据这一段语音设置为 0
            if ( res == 0 )
            {   
                for ( int j=0; j<m_vad_chunk_size; j++ )
                {
                    pdata[ i*m_vad_chunk_size + j ] = 0;
                }
                silence_counter += 1;
            }
        }

        logger << "Silence percent: " << silence_counter << "/" << N;
        logger.debug_s();

        return silence_counter == N ? true : false ;

        // std::cout << "active: " << debug_a_counter << " silent: " << debug_u_counter << std::endl;

        // 处理最后一段长度不足 m_vad_chunk_size 的语音
        // if ( N * m_vad_chunk_size < data_len_samples )
        // {
        //     int tail_length = data_len_samples - N * m_vad_chunk_size;
        //     for ( int j=0; j<data_len_samples; j++ )
        //     {
        //         buffer[ j ] = j < tail_length ? pdata[ i*m_vad_chunk_size + j ] : 0 ;
        //     }  
        //     int res = m_voice_detector->IsSpeech( buffer, m_vad_chunk_size, m_sample_rate );
        //     if (ret == Vad::kError) {
        //         logger.error("VAD process failed. Set it to: activaty!");
        //         ret = 1;
        //     }
        //     if ( ret == 0 )
        //     {
        //         for ( int j=0; j<tail_length; j++ )
        //         {
        //             pdata[ i*m_vad_chunk_size + j ] = 0;
        //         }
        //     }                   
        // }

        // 为了避免遗漏，最后一段先暂时不做，只允许设置vad chunk size 为160或则320，当采样率为16000和8000的时候都能整除

    }
    #endif
    
    void Speech_Engine::kws_weakup_recognition(short *pdata, int data_len_samples, char *output_keyword)
    {
        // init 
        std::string output_str;
        
        // 准备数据和特征
        int ret = prepare_data_and_feature(pdata, data_len_samples);
        if (ret != 0)
        {
            return;
        }

        // 如果语音特征未装满容器，不进行唤醒和关键词检测
        if (m_init_audio_length_ms < m_storage_audio_length_ms)
            return;

        // 滑窗 kws weakup 检测
        if (not m_kws_weakup_bool)
        {
            // 检查kws的计数器, 如果计数器超过抑制时长，才做检测
            m_kws_weakup_counter_ms += m_receive_audio_length_ms;
            if ( m_kws_weakup_counter_ms < m_kws_weakup_suppression_counter_ms )
            {
                return;
            }
            m_kws_weakup_counter_ms = 0;

            bool find_kws_weakup_bool = run_kws_weakup();

            if (find_kws_weakup_bool)
            {
                m_kws_weakup_bool = true;

                logger.debug( "===============!!!!!!!!!!!!!!===============" );
                logger.debug( "********************************************" );
                logger.debug( "**" );
                logger.info( "** Device Weakup: \033[0;31m Weakup \033[0;39m" );
                logger.debug( "**" );
                logger.debug( "********************************************" );
                output_str += "Weakup ";

                #ifdef _OUTPUTWAVE
                output_recognized_wave();
                #endif

                #ifdef _AIMESSAGE
                std::string message_string;
                Json::Value message_json = create_message_json();
                json_to_string(message_json, message_string);
                ret = send_msg((void *)message_string.c_str(), (int)strlen(message_string.c_str()));
                if (ret != 0)
                {
                    logger.error("Send_message is not right.");
                }
                // std::cout << "** [Information:] message_string: \033[0;31m" << message_string << "\033[0;39m" << std::endl;
                #endif
            }
        }
        else
        {
            m_weakup_asr_counter_ms += m_receive_audio_length_ms;
            if (m_weakup_asr_counter_ms >= m_kws_weakup_asr_suppression_counter_ms)
            {
                m_weakup_asr_counter_ms = 0;
                m_kws_weakup_bool = false;
            }
        }

        // output
        sprintf(output_keyword, "%s", output_str.c_str());
        return;
    }

    void Speech_Engine::run_kws_asr(std::string &output_str)
    {
        // init 
        std::vector<int> result_id;

        // kws forward
        run_kws(&result_id);
        
        if (result_id.size())
        {
            // // show
            // std::cout << std::endl;
            // for (int i = 0; i < result_id.size(); i++)
            // {
            //     std::cout << "KWS result_id: "<<result_id[i] << std::endl;
            // }

            // asr forward
            run_asr_after_kws(&result_id, &output_str);
        }

        // // asr_duplicate_check
        // if (output_str.length() > 0) {
        //     output_str = asr_duplicate_check(output_str);
        // }
        return;
    }

    void Speech_Engine::run_asr(std::string &output_str, bool contorl_kws_bool)
    {
        if ((m_algorithem_id == 1) || (m_algorithem_id == 2) || (m_algorithem_id == 4)) 
        {
            run_asr_normal(output_str, contorl_kws_bool);
        }
        else if (m_algorithem_id == 6)
        {   
            // 常规方案：先 bpe，后 phoneme，output_str 逻辑修改后，需要做一些改进
            // run_asr_normal(output_str, contorl_kws_bool);

            // if (output_str.length() > 0) {
            //     std::cout << "Bpe Detect Command: " << output_str << std::endl;

            //     if (!contorl_kws_bool) {
            //         output_str = "";
            //         run_asr_second(output_str, contorl_kws_bool);
            //         std::cout << "Phoneme Detect Command: " << output_str << std::endl;
            //     }
            // }

            // //  方案：仅适用 phoneme 模型
            // run_asr_second(output_str, contorl_kws_bool);
            // std::cout << "Phoneme Detect Command: " << output_str << std::endl;

            //  方案：控制词 bpe 模型，报警词 phoneme 模型
            if (contorl_kws_bool) {
                run_asr_normal(output_str, contorl_kws_bool);
                logger.debug( std::string("Bpe Detect Command: ") + output_str );
            }
            else
            {
                run_asr_second(output_str, contorl_kws_bool);
                logger.debug( std::string("Phoneme Detect Command: ") + output_str );
            }
            

        }

        #ifndef _BUS
        // asr_duplicate_check
        if (output_str.length() > 0) {
            output_str = asr_duplicate_check(output_str);
        }
        #endif
        return;
    }

    void Speech_Engine::run_asr_normal(std::string &output_str, bool contorl_kws_bool)
    {
        // // init
        // cv::Mat asr_cnn_out = cv::Mat::zeros(m_asr_model->output_feature_time(), m_asr_model->output_feature_num(), CV_32SC1);

        // #ifdef _TESTTIME
        // unsigned long long time_begin, time_end;
        // TEST_TIME(time_begin);
        // #endif

        // // forward
        // int ret = m_asr_model->asr_forward(m_feature_data_container, &asr_cnn_out);
        // if (ret != 0)
        // {
        //     logger.error( "ASR Forward Failed." );
        //     return;
        // }
        // // std::cout << "[Information:] m_feature_data_container.rows: " << m_feature_data_container.rows << ", m_feature_data_container.cols: " << m_feature_data_container.cols << std::endl;
        // // ASR::show_mat_uchar(m_feature_data_container, m_feature_data_container.rows, m_feature_data_container.cols);
        // // std::cout << "[Information:] asr_cnn_out.rows: " << asr_cnn_out.rows << ", asr_cnn_out.cols: " << asr_cnn_out.cols << std::endl;
        // // ASR::show_mat_float(asr_cnn_out, asr_cnn_out.rows, asr_cnn_out.cols);

        // // debug: cpp & python 一致性测试
        // // // 保存数据
        // // cv::FileStorage fs("/usr/local/audio_data/demo_test/test_feature_cpp.xml", cv::FileStorage::WRITE);
        // // fs << "feature" << m_feature_data_container;
        // // fs.release();
        // // // 这里需要做一次显示 float 的矩阵数据的拷贝，不然结果保存错误，原因未知
        // // cv::Mat fs_mat = cv::Mat_<float>(asr_cnn_out.rows, asr_cnn_out.cols);
        // // for (int i = 0; i < asr_cnn_out.rows; i++) {
        // //     for (int j = 0; j < asr_cnn_out.cols; j++) {
        // //         fs_mat.at<float>(i, j) = asr_cnn_out.at<float>(i, j);
        // //     }
        // // }
        // // cv::FileStorage fs("/usr/local/audio_data/demo_test/test_cpp.xml", cv::FileStorage::WRITE);
        // // fs << "test" << fs_mat;
        // // fs.release();

        // // // 加载数据
        // // // cv::FileStorage fs("/usr/local/audio_data/demo_test/test_python.xml", cv::FileStorage::READ);
        // // // fs["test"] >> asr_cnn_out;
        // // // fs.release();
        
        // #ifdef _TESTTIME
        // TEST_TIME(time_end);
        // asr_normal_model_forward_time += TIME_DELTA(time_begin,time_end);
        // TEST_TIME(time_begin);
        // #endif

        // // decode 
        // if (m_asr_decode_id == 0)
        // {
        //     // std::cout << "[Information:] bestpath_decoder: " << std::endl;
        //     m_asr_decode->bestpath_decoder(asr_cnn_out);
        // }
        // else if (m_asr_decode_id == 1)
        // {
        //     // std::cout << "[Information:] beamsearch_decoder: " << std::endl;
        //     m_asr_decode->beamsearch_decoder(asr_cnn_out);
        // }
        // else {
        //     logger.error( "Unknown m_asr_decode_id" );
        //     return;
        // }

        // #ifdef _TESTTIME
        // TEST_TIME(time_end);
        // asr_normal_search_time += TIME_DELTA(time_begin,time_end);
        // TEST_TIME(time_begin);
        // #endif

        // if(m_language_id == 0) {
        //     // chinese
        //     // show result
        //     // m_asr_decode->show_result_id();
        //     // m_asr_decode->show_symbol();
        //     // m_asr_decode->output_symbol(&output_str);
        //     // match 
        //     m_asr_decode->match_keywords_with_cache();

        //     // output 
        //     m_asr_decode->output_control_result_string_chinese(&output_str, contorl_kws_bool);
        //     // m_asr_decode->output_symbol(&output_str);
        //     // m_asr_decode->output_result_string(&output_str);
        // }
        // else if (m_language_id == 1) {
        //     // english
        //     // show result
        //     // m_asr_decode->show_result_id();
        //     // m_asr_decode->show_symbol();
        //     // m_asr_decode->show_symbol_english();

        //     // match 
        //     // bpe
        //     m_asr_decode->match_keywords_english_bpe();

        //     // phoneme
        //     // m_asr_decode->match_keywords_english_phoneme(contorl_kws_bool);

        //     // output 
        //     m_asr_decode->output_control_result_string(&output_str, contorl_kws_bool);
        //     // m_asr_decode->output_symbol(&output_str);
        //     // m_asr_decode->output_symbol_english(&output_str);
        //     // m_asr_decode->output_result_string(&output_str);
        // }
        // else {
        //     logger.error( "Unknown m_language_id" );
        //     return;
        // }

        // #ifdef _TESTTIME
        // TEST_TIME(time_end);
        // asr_normal_match_time += TIME_DELTA(time_begin,time_end);
        // #endif

        return;
    }

    void Speech_Engine::run_asr_second(std::string &output_str, bool contorl_kws_bool)
    {
        // // init
        // cv::Mat asr_cnn_out = cv::Mat::zeros(m_asr_second_model->output_feature_time(), m_asr_second_model->output_feature_num(), CV_32SC1);

        // #ifdef _TESTTIME
        // unsigned long long time_begin, time_end;
        // TEST_TIME(time_begin);
        // #endif

        // // forward
        // int ret = m_asr_second_model->asr_forward(m_feature_data_container, &asr_cnn_out);
        // if (ret != 0)
        // {
        //     logger.error( "ASR Forward Failed." );
        //     return;
        // }
        // // std::cout << "[Information:] m_feature_data_container.rows: " << m_feature_data_container.rows << ", m_feature_data_container.cols: " << m_feature_data_container.cols << std::endl;
        // // ASR::show_mat_uchar(m_feature_data_container, m_feature_data_container.rows, m_feature_data_container.cols);
        // // std::cout << "[Information:] asr_cnn_out.rows: " << asr_cnn_out.rows << ", asr_cnn_out.cols: " << asr_cnn_out.cols << std::endl;
        // // ASR::show_mat_float(asr_cnn_out, asr_cnn_out.rows, asr_cnn_out.cols);
        
        // #ifdef _TESTTIME
        // TEST_TIME(time_end);
        // asr_second_model_forward_time += TIME_DELTA(time_begin,time_end);
        // TEST_TIME(time_begin);
        // #endif

        // // decode 
        // if (m_asr_second_decode_id == 0)
        // {
        //     m_asr_second_decode->bestpath_decoder(asr_cnn_out);
        // }
        // else if (m_asr_second_decode_id == 1)
        // {
        //     m_asr_second_decode->beamsearch_decoder(asr_cnn_out);
        // }
        // else {
        //     logger.error( "Unknow m_asr_second_decode_id." );
        //     return;
        // }

        // #ifdef _TESTTIME
        // TEST_TIME(time_end);
        // asr_second_search_time += TIME_DELTA(time_begin,time_end);
        // TEST_TIME(time_begin);
        // #endif     

        // if(m_language_id == 0) {
        //     // chinese
        //     // m_asr_second_decode->show_result_id();
        //     // m_asr_second_decode->show_symbol();

        //     // m_asr_second_decode->output_symbol(&output_str);

        //     m_asr_second_decode->match_keywords_with_cache();

        //     m_asr_second_decode->output_control_result_string_chinese(&output_str, contorl_kws_bool);
        //     // m_asr_second_decode->output_result_string(&output_str);
        //     // m_asr_second_decode->show_result_string();
        // }
        // else if (m_language_id == 1) {
        //     // english
        //     // m_asr_second_decode->show_result_id();
        //     // m_asr_second_decode->show_symbol();
        //     // m_asr_second_decode->show_symbol_english();

        //     // m_asr_second_decode->output_symbol(&output_str);
        //     // m_asr_second_decode->output_symbol_english(&output_str);

        //     // bpe
        //     // m_asr_second_decode->match_keywords_english_bpe();

        //     // phoneme
        //     // m_asr_second_decode->match_keywords_english_phoneme(contorl_kws_bool);
        //     m_asr_second_decode->match_keywords_english_phoneme_simple(contorl_kws_bool);

        //     m_asr_second_decode->output_control_result_string(&output_str, contorl_kws_bool);
        //     // m_asr_second_decode->show_result_string();
        //     // m_asr_second_decode->output_result_string(&output_str);
        // }
        // else {
        //     logger.error( "Unknow m_language_id." );
        //     return;
        // }

        // #ifdef _TESTTIME
        // TEST_TIME(time_end);
        // asr_second_match_time += TIME_DELTA(time_begin,time_end);
        // #endif

        return;
    }

    bool Speech_Engine::run_kws_weakup()
    {
        bool find_kws_weekup = false;

        return find_kws_weekup;
    }

    void Speech_Engine::run_kws(std::vector<int> *result_id)
    {
        return;
    }

    void Speech_Engine::run_asr_after_kws(std::vector<int> *kws_id, std::string *output_str)
    {
        return;
    }

    #ifdef _OUTPUTWAVE
    void Speech_Engine::output_recognized_wave(std::string output_prefix_name, std::string output_folder)
    {   
        if(access(output_folder.c_str(), 0) == -1)
            return;

        // test: output wave
        time_t timep = time(0);
        char timec[64];
        ASR::Wave_Data wave_data;

        strftime(timec, sizeof(timec), "%Y-%m-%d-%H%M%S", localtime(&timep));

        signed short int output_data_container[m_output_recognized_data_container_samples];
        for (int i = 0; i < m_output_recognized_data_container_samples; i++)
        {
            output_data_container[i] = m_output_recognized_data_container[i];
            // std::cout << output_data_container[i]<< " ";
        }
        // std::cout << std::endl;

        std::string timec_string = timec;
        // amba
        #ifdef _AMBA
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_recognized_wave_id) + ".wav";
        #endif

        // novt
        #ifdef _NOVT
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_recognized_wave_id) + ".wav";
        #endif

        // hisi
        #ifdef _HISI
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_recognized_wave_id) + ".wav";
        #endif

        #ifdef _NCNN
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_recognized_wave_id) + ".wav";
        #endif

        wave_data.write_data(output_data_container, m_output_recognized_data_container_samples, m_sample_rate, output_wave_path.c_str());
        m_speech_options.output_recognized_wave_id ++;
    }

    void Speech_Engine::output_weakup_wave(std::string output_prefix_name, std::string output_folder)
    {
        if(access(output_folder.c_str(), 0) == -1)
            return;

        // test: output wave
        time_t timep = time(0);
        char timec[64];
        ASR::Wave_Data wave_data;

        strftime(timec, sizeof(timec), "%Y-%m-%d-%H%M%S", localtime(&timep));

        signed short int output_data_container[m_output_weakup_data_container_samples];
        for (int i = 0; i < m_output_weakup_data_container_samples; i++)
        {
            output_data_container[i] = m_output_weakup_data_container[i];
            // std::cout << output_data_container[i]<< " ";
        }
        // std::cout << std::endl;

        std::string timec_string = timec;
        // amba
        #ifdef _AMBA
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_weakup_wave_id) + ".wav";
        #endif

        // novt
        #ifdef _NOVT
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_weakup_wave_id) + ".wav";
        #endif

        // hisi
        #ifdef _HISI
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_weakup_wave_id) + ".wav";
        #endif

        #ifdef _NCNN
        std::string output_wave_path = output_folder + output_prefix_name + "_" + timec_string + "_" + std::to_string(m_speech_options.output_weakup_wave_id) + ".wav";
        #endif

        wave_data.write_data(output_data_container, m_output_weakup_data_container_samples, m_sample_rate, output_wave_path.c_str());
        m_speech_options.output_weakup_wave_id ++;
    }
    #endif

    void Speech_Engine::asr_duplicate_update_counter() {
        for (std::map<std::string, int>::iterator it = m_asr_duplicate_counter_map.begin(); it != m_asr_duplicate_counter_map.end(); ++it) 
            {  
                if (it->second > 0) {
                    it->second = it->second - m_receive_audio_length_ms;
                    // std::cout << it->first << ", " << it->second << std::endl; 
                }
            }
    }

    std::string Speech_Engine::asr_duplicate_check(std::string output_str) {
        std::string res_string = "";
        std::vector<std::string> tmp_string = ASR::StringSplit(output_str, " ");
        for(unsigned int i = 0; i < tmp_string.size(); i++)
        {
            std::map<std::string, int>::iterator it = m_asr_duplicate_counter_map.find(tmp_string[i]);
            if (it == m_asr_duplicate_counter_map.end()){
                m_asr_duplicate_counter_map[tmp_string[i]] = m_storage_audio_length_ms;
                res_string.append(tmp_string[i]);
                res_string.append(" ");
            } else {
                if (it->second > 0) 
                    continue;
                else {
                    it->second = m_storage_audio_length_ms;
                    res_string.append(tmp_string[i]);
                    res_string.append(" ");
                }
            }
        }
        return res_string;
    }

} // namespace ASR