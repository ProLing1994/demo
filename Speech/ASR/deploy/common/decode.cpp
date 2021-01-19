#include <algorithm>

#include "decode.hpp"

namespace ASR
{
    Decode::Decode()
    {
    }

    Decode::Decode(const int blank_id)
    {
        m_blank_id = blank_id;
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

    int Decode::get_edit_dist(std::vector<std::string> string1, std::vector<std::string> string2)
    {
        int m = string1.size();
        int n = string2.size();
        cv::Mat v = cv::Mat::zeros(m + 1, n + 1, CV_32SC1);
        for (int i = 0; i < m; i++)
        {
            v.at<int>(i + 1, 0) = i + 1;
        }
        for (int j = 0; j < n; j++)
        {
            v.at<int>(0, j + 1) = j + 1;
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (string1[i] == string2[j])
                {
                    v.at<int>(i + 1, j + 1) = v.at<int>(i, j);
                }
                else
                {
                    v.at<int>(i + 1, j + 1) = 1 + std::min(std::min(v.at<int>(i + 1, j), v.at<int>(i, j + 1)), v.at<int>(i, j));
                }
            }
        }
        return v.at<int>(m, n);
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