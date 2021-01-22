#include "model.hpp"

namespace ASR
{
    Model::Model()
    {
        // amba_init();

        // m_src_image = nullptr;
        // m_asr_model = nullptr;
    }

    Model::Model(const Model_Options_S &model_options)
    {
        m_model_options = Model_Options_S(model_options);
        
        // amba_init();

        // m_src_image = nullptr;
        // m_asr_model = nullptr;
    }

    Model::~Model()
    {
        // if(m_src_image != nullptr)
        // {
        //     amb_cv2x::DestroyImage(&m_src_image);
        // }
        
        // if(m_asr_model != nullptr)
        // {
        //     delete ((amb_cv2x::CNN *)(m_asr_model));
        // }
    }

    // void Model::amba_init()
    // {
    //     amb_cv2x::InitImageModule();
    //     memset((void *)&m_cnn_param, 0, sizeof(CV2X_CNN_PARAM_S));
    // }

    // int Model::amba_net_init(const char *model_path, void **model, const char *out_port_name, int rgb_type)
    // {
    //     //========initialize param========
    //     memset((void *)&m_cnn_param, 0, sizeof(CV2X_CNN_PARAM_S));

    //     strcpy(m_cnn_param.cavalry_bin, model_path);
    //     memset(m_cnn_param.mean_name, '\0', CV2X_FILENAME_LENGTH);
    //     strcpy(m_cnn_param.out_port_name[0], "conv7");
    //     memset(m_cnn_param.out_port_name[1], '\0', CV2X_FILENAME_LENGTH);
    //     m_cnn_param.rgb_type = 2;
    //     *model = (void *)(new amb_cv2x::CNN(&m_cnn_param));
    //     return 0;
    // }
    
    int Model::asr_init(const char *model_path, int feature_freq, int feature_time, const char *out_port_name, int rgb_type)
    {
        // // image 
        // m_src_image = amb_cv2x::CreateImage(CV2X_IMAGE_TYPE_U8C1, feature_freq, feature_time);

        // // model
        // amba_net_init(model_path, &m_asr_model, out_port_name, rgb_type);
        // return 0;
    }

    int Model::asr_forward(cv::Mat &input, cv::Mat *output)
    {   
        // // check
        // if(input.rows !=  m_model_options.input_feature_time or input.cols !=  m_model_options.input_feature_freq)
        // {
        //     printf("[ERROR:] %s, %d: Wrong Input Feature Shape.\n", __FUNCTION__, __LINE__);
        // }
        // if(output->rows !=  m_model_options.output_feature_time or output->cols !=  m_model_options.output_feature_num)
        // {
        //     printf("[ERROR:] %s, %d: Wrong Output Shape.\n", __FUNCTION__, __LINE__);
        // }

        // // input
        // input.copyTo(cv::Mat(cv::Size(m_src_image->u32Width, m_src_image->u32Height), CV_8UC1, (unsigned char*)(m_src_image->au64VirAddr[0]), m_src_image->au32Stride[0])); 

        // // forward
        // std::vector<CV2X_BLOB_S> layerout;
        // int ret = ((amb_cv2x::CNN *)(m_asr_model))->Run(m_src_image, layerout);
        // if(layerout.size() == 0){
        //     printf("[ERROR:] %s, %d: ASR Net Forward failed.\n", __FUNCTION__, __LINE__);
        //     return ret;
        // }
        // // std::cout << "\033[0;31m" << "[Information:]layerout.size: " << layerout.size()  <<"\033[0;39m" << std::endl;

        // // output
        // for(int i = 0; i < layerout.size(); i++)
        // {
        //     unsigned int u32Width = layerout[i].unShape.stWhc.u32Width;
        //     unsigned int u32Height = layerout[i].unShape.stWhc.u32Height;
        //     unsigned int u32Chn = layerout[i].unShape.stWhc.u32Chn;
        //     unsigned int u32Stride = layerout[i].u32Stride;
        //     // std::cout << "\033[0;31m" << "[Information:] u32Chn: " << u32Chn << ", u32Height: " << u32Height \
        //     //         << ", u32Width: " << u32Width  << ", u32Stride: " << u32Stride << "\033[0;39m" << std::endl;

        //     float *pResultBlob = (float *)(layerout[i].u64VirAddr);

        //     for (int i = 0; i < u32Chn; i++)
        //     {
        //         for (unsigned int j = 0; j < u32Height; j++)
        //         {
        //             for (unsigned int k = 0; k < u32Width; k++)
        //             {
        //                 // std::cout << "i: " << i << ", j: " << j <<  ", k: " << k << ", value: "<< *(pResultBlob + k) << std::endl;
        //                 output->at<float>(j, i) = *(pResultBlob + k);
        //             }
        //             // std::cout << "u32Stride / sizeof(float): " << u32Stride / sizeof(float) << std::endl;
        //             pResultBlob += u32Stride / sizeof(float);
        //         }
        //     }

        // }
        // printf("[Information:] Model Forward Done!!!\n");
        // return ret;
    }

} // namespace ASR