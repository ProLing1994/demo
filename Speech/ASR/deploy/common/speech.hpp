#ifndef _ASR_SPEECH_H_
#define _ASR_SPEECH_H_

#include <memory>
#include <vector>
#include <string>

#include "decode.hpp"
#include "feature.hpp"

namespace ASR
{
    struct ASR_PARAM_S
	{
		ASR_PARAM_S() : n_fft(512),
						time_seg_ms(32),
						time_step_ms(10),
						feature_freq(48),
						feature_time(296),
                        fc1_kernels(4096),
						fc2_kernels(256),
						output_num(19),
						sample_rate(16000),
						algorithem_id(0) {}

		int n_fft;
		int time_seg_ms;
		int time_step_ms;
		int feature_freq;
		int feature_time;
        int fc1_kernels;
		int fc2_kernels;
		int output_num;
		int sample_rate;
		int algorithem_id;
	};

    class Speech_Engine
	{
    public:
        Speech_Engine();
        Speech_Engine(const ASR_PARAM_S &params, const char *file_path = NULL);
        ~Speech_Engine();
    
    public:
        void load_param(const ASR_PARAM_S &params, const char *file_path = NULL);
        void speech_recognition(short *pdata, int data_len_samples, char *outKeyword);

	private:
        void asr_recognition(short *pdata, int data_len_samples, char *outKeyword);
        void kws_asr_recognition(short *pdata, int data_len_samples, char *outKeyword);

    private:
        std::unique_ptr<Feature> m_feature;
        std::unique_ptr<Decode> m_decode;
        
        int m_algorithem_id;
        int m_output_num;
		int m_fc1_dim;
		int m_fc2_dim;

        std::vector<std::string> m_keywords;
        std::vector<std::string> m_symbollist;
        std::vector<std::string> m_hanzi_kws_list;
		std::vector<std::vector<std::string>> m_pny_kws_list;
        
    };

} // namespace ASR

#endif // _ASR_SPEECH_H_