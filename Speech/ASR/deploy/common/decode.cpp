#include "decode.hpp"

namespace ASR
{
    Decode::Decode()
    {
    }
    
    Decode::~Decode()
    {
    }

    void Decode::ctc_decoder(float *input, int rows, int cols, bool greedy)
    {
        // init
        int frame_num = rows;
        int feature_num = cols;

        int last_max_id = 0;
        if (greedy)
        {
            float max_value = 0;
            int max_id = 0;
            for (int i = 0; i < frame_num; i++)
            {
                max_value = input[i * feature_num + 0];
                max_id = 0;
                // if (i == 0) std::cout << i * feature_num + 0 << ": " << input[i * feature_num + 0] << std::endl;
                for (int j = 1; j < feature_num; j++)
                {   
                    // if (i == 0)  std::cout << i * feature_num + j << ": " <<  input[i * feature_num + j] << std::endl;                    
                    if (input[i * feature_num + j] > max_value)
                    {
                        max_value = input[i * feature_num + j] ;
                        max_id = j;
                    }
                }
                if (max_id != (m_blank_id) && last_max_id != max_id)
                {
                    m_result_id.push_back(max_id);
                    last_max_id = max_id;
                }
            }
        }
        return;
    }

    void Decode::ctc_decoder(cv::Mat input, bool greedy)
    {
        int frame_num = input.rows;
        int feature_num = input.cols;

        int last_max_id = 0;
        if (greedy)
        {
            float max_value = 0;
            int max_id = 0;
            for (int i = 0; i < frame_num; i++)
            {
                max_value = input.at<int>(i, 0);
                max_id = 0;
                for (int j = 1; j < feature_num; j++)
                {
                    if (input.at<int>(i, j) > max_value)
                    {
                        max_value = input.at<int>(i, j);
                        max_id = j;
                    }
                }
                if (max_id != (m_blank_id) && last_max_id != max_id)
                {
                    m_result_id.push_back(max_id);
                    last_max_id = max_id;
                }
            }
        }

        return;
    }

    void Decode::copy_result_id_to(int *data)
    {
        int *temp_data = new int[m_result_id.size()];
        for(int i = 0; i < m_result_id.size(); i++)
        {
            temp_data[i] = m_result_id[i];
        }
        memcpy(data, temp_data, m_result_id.size() * sizeof(int));
    }

} // namespace ASR