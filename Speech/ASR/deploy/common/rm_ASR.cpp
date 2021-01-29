#include "config.h"
#include "rm_ASR.hpp"
#include "speech.hpp"

#define ASR_OPEN_API __attribute__((visibility("default")))

static ASR::Speech_Engine *g_asr_engine;

int load_params(char *file_path, ASR::ASR_PARAM_S &params)
{
    char config_file[256] = {0};
	sprintf(config_file, "%s/configFiles/configFileASR.cfg", file_path);
	rmCConfig cconfig;
	int ret = cconfig.OpenFile(config_file);
	if (ret != 0)
	{
		printf("[ERROR:] %s, %d: Read Config Failed.\n", __FUNCTION__, __LINE__);
		return ret;
	}

	params.algorithem_id = cconfig.GetInt("AsrAlgorithem_ID", "algorithem_id");
	if (params.algorithem_id == 0)
	{
		params.n_fft = cconfig.GetInt("KwsParam", "n_fft");
		params.time_seg_ms = cconfig.GetInt("KwsParam", "time_seg_ms");
		params.time_step_ms = cconfig.GetInt("KwsParam", "time_step_ms");
		params.output_num = cconfig.GetInt("KwsParam", "output_num");
		params.feature_freq = cconfig.GetInt("KwsParam", "feature_freq");
		params.feature_time = cconfig.GetInt("KwsParam", "feature_time");
		params.sample_rate = cconfig.GetInt("KwsParam", "fs");
	}
	else
	{
		params.n_fft = cconfig.GetInt("AsrParam", "n_fft");
		params.time_seg_ms = cconfig.GetInt("AsrParam", "time_seg_ms");
		params.time_step_ms = cconfig.GetInt("AsrParam", "time_step_ms");
		params.output_num = cconfig.GetInt("AsrParam", "output_num");
		params.output_time = cconfig.GetInt("AsrParam", "output_time");
		params.feature_freq = cconfig.GetInt("AsrParam", "feature_freq");
		params.feature_time = cconfig.GetInt("AsrParam", "feature_time");
		params.model_name = cconfig.GetStr("AsrParam", "model_name");
		params.output_name = cconfig.GetStr("AsrParam", "output_name");
		params.sample_rate = cconfig.GetInt("AsrParam", "fs");
	}

    return ret;
}

ASR_OPEN_API int RMAPI_AI_AsrInit(char *file_path)
{
    ASR::ASR_PARAM_S params;
    int ret = load_params(file_path, params);
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
	g_asr_engine->speech_recognition(pBuffer, length, outKeyword);
	return 0;
}
