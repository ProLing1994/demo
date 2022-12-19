#include "asr_config.h"
#include "rm_ASR.hpp"
#include "speech.hpp"
#include "common.hpp"

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
extern double vad_time;
#endif 

extern double feature_time;

extern double kws_weakup_model_forward_time;
extern double kws_weakup_decode_time;

extern double asr_normal_model_forward_time;
extern double asr_normal_search_time;    
extern double asr_normal_match_time;    

extern double asr_second_model_forward_time;
extern double asr_second_search_time;
extern double asr_second_match_time;

#endif

#define ASR_OPEN_API __attribute__((visibility("default")))

static ASR::Speech_Engine *g_asr_engine;
static ASR::Logger logger = ASR::getLogger();

void load_normal_params(rmCConfig &cconfig, ASR::ASR_PARAM_S &params)
{
	params.algorithem_id = cconfig.GetInt("AsrAlgorithem_ID", "algorithem_id");
	params.language_id = cconfig.GetInt("AsrAlgorithem_ID", "language_id");
	params.streamanager_id = cconfig.GetInt("AsrAlgorithem_ID", "streamanager_id");
	params.description = cconfig.GetStr("AsrAlgorithem_ID", "description");
	if(params.algorithem_id==0)
	{
		params.n_fft = cconfig.GetInt("NormalParam_FFT", "n_fft");
		params.nfilt = cconfig.GetInt("NormalParam", "nfilt");
		params.sample_rate = cconfig.GetInt("NormalParam_FFT", "fs");
		params.receive_audio_length_s = cconfig.GetFloat("NormalParam_FFT", "receive_audio_length_s");
		params.time_seg_ms = cconfig.GetInt("NormalParam_FFT", "time_seg_ms");
		params.time_step_ms = cconfig.GetInt("NormalParam_FFT", "time_step_ms");
		params.feature_freq = cconfig.GetInt("NormalParam_FFT", "feature_freq");
		params.feature_time = cconfig.GetInt("NormalParam_FFT", "feature_time");
		params.storage_audio_length_s = cconfig.GetInt("NormalParam_FFT", "storage_audio_length_s");
		params.storage_feature_time = cconfig.GetInt("NormalParam_FFT", "storage_feature_time");		
	}
	else
	{
		params.n_fft = cconfig.GetInt("NormalParam", "n_fft");
		params.nfilt = cconfig.GetInt("NormalParam", "nfilt");
		params.sample_rate = cconfig.GetInt("NormalParam", "fs");
		params.receive_audio_length_s = cconfig.GetFloat("NormalParam", "receive_audio_length_s");
		params.time_seg_ms = cconfig.GetInt("NormalParam", "time_seg_ms");
		params.time_step_ms = cconfig.GetInt("NormalParam", "time_step_ms");
		params.feature_freq = cconfig.GetInt("NormalParam", "feature_freq");
		params.feature_time = cconfig.GetInt("NormalParam", "feature_time");
		params.storage_audio_length_s = cconfig.GetInt("NormalParam", "storage_audio_length_s");
		params.storage_feature_time = cconfig.GetInt("NormalParam", "storage_feature_time");
	}
	return;
}


void load_asr_params(rmCConfig &cconfig, ASR::ASR_PARAM_S &params, std::string mAttr="AsrParam")
{
	params.asr_model_name = cconfig.GetStr(mAttr.c_str(), "model_name");
	params.asr_param_name = cconfig.GetStr(mAttr.c_str(), "param_name");
	params.asr_input_name = cconfig.GetStr(mAttr.c_str(), "input_name");
	params.asr_output_name = cconfig.GetStr(mAttr.c_str(), "output_name");
	params.asr_feature_freq = cconfig.GetInt(mAttr.c_str(), "feature_freq");
	params.asr_feature_time = cconfig.GetInt(mAttr.c_str(), "feature_time");
	params.asr_output_num = cconfig.GetInt(mAttr.c_str(), "output_num");
	params.asr_output_time = cconfig.GetInt(mAttr.c_str(), "output_time");
	params.asr_suppression_counter = cconfig.GetInt(mAttr.c_str(), "asr_suppression_counter");
	params.asr_decode_id = cconfig.GetInt(mAttr.c_str(), "decode_id");

	params.asr_dict_path = cconfig.GetStr(mAttr.c_str(), "dict_path");
	params.asr_lm_path = cconfig.GetStr(mAttr.c_str(), "lm_path");
	params.asr_lm_ngram_length = cconfig.GetInt(mAttr.c_str(), "lm_ngram_length");
	return;
}

void load_asr_second_params(rmCConfig &cconfig, ASR::ASR_PARAM_S &params, std::string mAttr="AsrPhonemeParam")
{
	params.asr_second_model_name = cconfig.GetStr(mAttr.c_str(), "model_name");
	params.asr_second_param_name = cconfig.GetStr(mAttr.c_str(), "param_name");
	params.asr_second_input_name = cconfig.GetStr(mAttr.c_str(), "input_name");
	params.asr_second_output_name = cconfig.GetStr(mAttr.c_str(), "output_name");
	params.asr_second_output_num = cconfig.GetInt(mAttr.c_str(), "output_num");
	params.asr_second_output_time = cconfig.GetInt(mAttr.c_str(), "output_time");
	params.asr_second_decode_id = cconfig.GetInt(mAttr.c_str(), "decode_id");

	params.asr_second_dict_path = cconfig.GetStr(mAttr.c_str(), "dict_path");
	params.asr_second_lm_path = cconfig.GetStr(mAttr.c_str(), "lm_path");
	params.asr_second_lm_ngram_length = cconfig.GetInt(mAttr.c_str(), "lm_ngram_length");
	return;
}

void load_kws_weakup_params(rmCConfig &cconfig, ASR::ASR_PARAM_S &params)
{
	params.kws_weakup_model_name = cconfig.GetStr("WeakupParam", "model_name");
	params.kws_weakup_param_name = cconfig.GetStr("WeakupParam", "param_name");
	params.kws_weakup_input_name = cconfig.GetStr("WeakupParam", "input_name");
	params.kws_weakup_output_name = cconfig.GetStr("WeakupParam", "output_name");
	params.kws_weakup_feature_freq = cconfig.GetInt("WeakupParam", "feature_freq");
	params.kws_weakup_feature_time = cconfig.GetInt("WeakupParam", "feature_time");
	params.kws_weakup_output_num = cconfig.GetInt("WeakupParam", "output_num");
	params.kws_weakup_output_time = cconfig.GetInt("WeakupParam", "output_time");
	params.kws_weakup_threshold = cconfig.GetFloat("WeakupParam", "weakup_threshold");
	params.kws_weakup_number_threshold = cconfig.GetFloat("WeakupParam", "weakup_number_threshold");
	params.kws_weakup_suppression_counter = cconfig.GetInt("WeakupParam", "weakup_suppression_counter");
	params.kws_weakup_asr_suppression_counter = cconfig.GetInt("WeakupParam", "weakup_asr_suppression_counter");

	#ifdef _BUS
	params.kws_weakup_activity_1st_counter = cconfig.GetInt("ExtraWeakupParamForBus", "weakup_activity_1st_counter");
	params.kws_weakup_activity_2nd_counter = cconfig.GetInt("ExtraWeakupParamForBus", "weakup_activity_2nd_counter");
	params.kws_weakup_shield_counter = cconfig.GetInt("ExtraWeakupParamForBus", "weakup_shield_counter");
	params.kws_weakup_remove_tts = cconfig.GetBool("ExtraWeakupParamForBus", "weakup_remove_tts");
	params.vad_mode = cconfig.GetStr("ExtraWeakupParamForBus", "vad_mode");
	#endif
	
	return;
}

void load_kws_params(rmCConfig &cconfig, ASR::ASR_PARAM_S &params)
{
	params.kws_model_name = cconfig.GetStr("KwsParam", "model_name");
	params.kws_param_name = cconfig.GetStr("KwsParam", "param_name");
	params.kws_input_name = cconfig.GetStr("KwsParam", "input_name");
	params.kws_output_name = cconfig.GetStr("KwsParam", "output_name");
	params.kws_feature_freq = cconfig.GetInt("KwsParam", "feature_freq");
	params.kws_feature_time = cconfig.GetInt("KwsParam", "feature_time");
	params.kws_output_num = cconfig.GetInt("KwsParam", "output_num");
	params.kws_output_time = cconfig.GetInt("KwsParam", "output_time");
	
	return;
}

int load_params(char *file_path, ASR::ASR_PARAM_S &params)
{
    char config_file[256] = {0};
	sprintf(config_file, "%s/kws/configFiles/configFileASR.cfg", file_path);
	rmCConfig cconfig;
	int ret = cconfig.OpenFile(config_file);
	if (ret != 0)
	{
		logger.error("Read Config Failed.");
		return ret;
	}
	load_normal_params(cconfig, params);
	if (params.algorithem_id == 0)
	{
		load_asr_params(cconfig, params);
		load_kws_params(cconfig, params);

		// check
		if (params.feature_freq != params.asr_feature_freq || params.feature_freq != params.kws_feature_freq)
		{
			printf("[ERROR:] %s, %d: Unknow feature_freq.\n", __FUNCTION__, __LINE__);
			return -1;
		}
		if (params.kws_feature_time > params.asr_feature_time)
		{
			printf("[ERROR:] %s, %d: Unknow feature_time.\n", __FUNCTION__, __LINE__);
			return -1;
		}
	}
	else if ((params.algorithem_id == 1) || (params.algorithem_id == 2)) 
	{
		load_asr_params(cconfig, params);

		// check
		if (params.feature_freq != params.asr_feature_freq)
		{
			logger.error("Unknow feature_freq.");
			return -1;
		}
	}
	else if (params.algorithem_id == 4)
	{
		load_asr_params(cconfig, params);
		load_kws_weakup_params(cconfig, params);

		// check
		if (params.feature_freq != params.asr_feature_freq || params.feature_freq != params.kws_weakup_feature_freq)
		{
			logger.error("Unknow feature_freq.");
			return -1;
		}
		if (params.kws_weakup_feature_time > params.asr_feature_time)
		{
			logger.error("Unknow feature_time.");
			return -1;
		}
	}
	else if ((params.algorithem_id == 3) || (params.algorithem_id == 5))
	{
		load_kws_weakup_params(cconfig, params);

		// check
		if (params.feature_freq != params.kws_weakup_feature_freq)
		{
			logger.error("Unknow feature_freq.");
			return -1;
		}
	}
	else if (params.algorithem_id == 6)
	{
		load_asr_params(cconfig, params, "AsrParam");
		load_asr_second_params(cconfig, params, "AsrSecondParam");
		load_kws_weakup_params(cconfig, params);

		// check
		if (params.feature_freq != params.asr_feature_freq || params.feature_freq != params.kws_weakup_feature_freq)
		{
			logger.error("Unknow feature_freq.");
			return -1;
		}
		if (params.kws_weakup_feature_time > params.asr_feature_time)
		{
			logger.error("Unknow feature_time.");
			return -1;
		}
	}
	else {
		logger.error("Unknow m_algorithem_id.");
		return -1;
	}

    return 0;
}


void show_update_info(ASR::ASR_PARAM_S & params)
{

	logger.info( "*************** ASR ALGORITHM UPDATE INFO ***************" );
	logger.info("**");
	// 1, 显示算法库生成时间
	// logger << "** DATETIME: " << TIMESTAMP;
	logger.info_s();
	if ( params.description.size() > 0 ) 
	{ 
		logger << "** DESCRIPTION: " << params.description; 
		logger.info_s();
	}
	
	// 2. 显示一些参数
	if (params.algorithem_id == 0)
	{
		logger << "** KWS MODEL: " << params.kws_model_name;
		logger.info_s();
		logger << "** ASR MODEL: " << params.asr_model_name;
		logger.info_s();
	}
	else if (params.algorithem_id == 1 || params.algorithem_id == 2)
	{
		logger << "** ASR MODEL: " << params.asr_model_name;
		logger.info_s();
	}
	else if (params.algorithem_id == 4)
	{
		logger << "** KWS MODEL: " << params.kws_weakup_model_name;
		logger.info_s();
		logger << "** ASR MODEL: " << params.asr_model_name;
		logger.info_s();
	}
	else if (params.algorithem_id == 3 || params.algorithem_id == 5)
	{
		logger << "** KWS MODEL: " << params.kws_weakup_model_name;
		logger.info_s();
	}
	else if (params.algorithem_id == 6)
	{
		logger << "** KWS MODEL: " << params.kws_weakup_model_name;
		logger.info_s();
		logger << "** ASR 1ST MODEL: " << params.asr_model_name;
		logger.info_s();
		logger << "** ASR 2ND MODEL: " << params.asr_second_model_name;
		logger.info_s();
	}
	logger.info("**");
	logger.info( "*********************************************************" );
}

ASR_OPEN_API int RMAPI_AI_AsrInit(char *file_path)
{
	//logger.info( "ASR algorithm update date: 2021/08/18 A." );
	
    ASR::ASR_PARAM_S params;
    int ret = load_params(file_path, params);
	show_update_info(params);
	g_asr_engine = new ASR::Speech_Engine(params, file_path);
    return ret;
}

ASR_OPEN_API int RMAPI_AI_AsrDeinit()
{
	delete g_asr_engine;
	g_asr_engine = nullptr;
	return 0;
}

ASR_OPEN_API int RMAPI_AI_AsrAlgStart(short *pBuffer, int length, char *outKeyword)
{
	#ifdef _TESTTIME
	unsigned long long begin, end;
	TEST_TIME(begin);
	#endif

	g_asr_engine->speech_recognition(pBuffer, length, outKeyword);
	
	#ifdef _TESTTIME
	TEST_TIME(end);
	double runtime = TIME_DELTA(begin,end);

	char runtimeinfo[256];
	#ifdef _BUS
	sprintf(runtimeinfo,"total: %.2f ms, vad: %.2f ms, feat: %.2f ms, kws: %.2f + %.2f ms, normal: %.2f + %.2f + %.2f ms, second: %.2f + %.2f + %.2f ms.",  \
						            runtime, vad_time, feature_time, \
								    kws_weakup_model_forward_time, kws_weakup_decode_time, \
								    asr_normal_model_forward_time, asr_normal_search_time, asr_normal_match_time, \
									asr_second_model_forward_time, asr_second_search_time, asr_second_match_time );
	vad_time = 0.0;
	#else
	sprintf(runtimeinfo,"total: %.2f ms, feat: %.2f ms, kws: %.2f + %.2f ms, normal: %.2f + %.2f + %.2f ms, second: %.2f + %.2f + %.2f ms.",  \
						            runtime, feature_time, \
								    kws_weakup_model_forward_time, kws_weakup_decode_time, \
								    asr_normal_model_forward_time, asr_normal_search_time, asr_normal_match_time, \
									asr_second_model_forward_time, asr_second_search_time, asr_second_match_time );
	#endif
	logger.debug( runtimeinfo );

	feature_time = 0.0;

	kws_weakup_model_forward_time = 0.0;
	kws_weakup_decode_time = 0.0;

	asr_normal_model_forward_time = 0.0;
	asr_normal_search_time = 0.0;    
	asr_normal_match_time = 0.0;    

	asr_second_model_forward_time = 0.0;
	asr_second_search_time = 0.0;
	asr_second_match_time = 0.0;	
	#endif
	
	return 0;
}

#ifdef _BUS
ASR_OPEN_API int RMAPI_AI_AsrAlgActivate()
{
	g_asr_engine->activate_weakup_forcely();
	return 0;
}

ASR_OPEN_API int RMAPI_AI_AsrAlgDeactivate()
{
	g_asr_engine->deactivate_weakup_forcely();
	return 0;
}
#endif