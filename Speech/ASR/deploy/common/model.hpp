#ifndef _ASR_MODEL_H_
#define _ASR_MODEL_H_

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

// #include "rmai_cv2x.hpp"

namespace ASR
{
    struct Model_Options_S
	{
		Model_Options_S(): 
            input_feature_freq(48),
            input_feature_time(296),
            output_feature_num(408),
            output_feature_time(35) {}

        Model_Options_S(const Model_Options_S &model_options): 
            input_feature_freq(model_options.input_feature_freq),
            input_feature_time(model_options.input_feature_time), 	
            output_feature_num(model_options.output_feature_num),
            output_feature_time(model_options.output_feature_time) {}

        int input_feature_freq;
        int input_feature_time;
        int output_feature_num;
        int output_feature_time;
	};

    class Model
    {
    public:
        Model();
        Model(const Model_Options_S &model_options);
        ~Model();

        inline int input_feature_freq() const { return m_model_options.input_feature_freq; } 
        inline int input_feature_time() const { return m_model_options.input_feature_time; } 
        inline int output_feature_num() const { return m_model_options.output_feature_num; }
        inline int output_feature_time() const { return m_model_options.output_feature_time; }

    public:
        int asr_init(const char *model_path, int feature_freq, int feature_time, const char *out_port_name="conv7", int rgb_type=2);
        int asr_forward(cv::Mat &input, cv::Mat *output);

    // private:
    //     void amba_init();
    //     int amba_net_init(const char *model_path, void **model, const char *out_port_name="conv7", int rgb_type=2);
    private:
        Model_Options_S m_model_options;

        // CV2X_CNN_PARAM_S m_cnn_param;

        // // image 
        // CV2X_IMAGE_S* m_src_image;
        
        // // model
        // void *m_asr_model;
        // // void *m_vad_model;
        // // void *m_kws_model;
        // // void *m_kws_cas_model;
    };
} // namespace ASR

#endif // _ASR_MODEL_H_