#include <algorithm>

#include "decode.hpp"

namespace ASR
{
    Decode::Decode()
    {
    }

    Decode::Decode(const std::vector<std::string> &keywords, const std::vector<std::string> &symbol_list)
    {
        m_keywords.reserve(keywords.size());
        for(int i = 0; i < keywords.size(); i++)
        {
            m_keywords.push_back(keywords[i]);
        }

        m_symbol_list.reserve(symbol_list.size());
        for(int i = 0; i < symbol_list.size(); i++)
        {
            m_symbol_list.push_back(symbol_list[i]);
        }
    }

    Decode::Decode(const int blank_id, const std::vector<std::string> &symbol_list, 
                    const std::vector<std::string> &hanzi_kws_list, const std::vector<std::vector<std::string>> &pinyin_kws_list)
    {
        m_blank_id = blank_id;

        m_symbol_list.reserve(symbol_list.size());
        for(int i = 0; i < symbol_list.size(); i++)
        {
            m_symbol_list.push_back(symbol_list[i]);
        }

        m_hanzi_kws_list.reserve(hanzi_kws_list.size());
        for(int i = 0; i < hanzi_kws_list.size(); i++)
        {
            m_hanzi_kws_list.push_back(hanzi_kws_list[i]);
        }

        m_pinyin_kws_list.reserve(pinyin_kws_list.size());
        for(int i = 0; i < pinyin_kws_list.size(); i++)
        {
            m_pinyin_kws_list.push_back(pinyin_kws_list[i]);
        }
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

    void Decode::match_keywords_robust(std::string *out)
    {
        int index = 0;
        int dist = 0;
        int out_index = 10000;
        bool match_flag = false;
        std::vector<int> match_id;
        std::vector<std::string> tmp_kws;
        std::vector<std::string> result_str;
        for (int i = 0; i < m_result_id.size(); i++)
        {
            result_str.push_back(m_symbol_list[m_result_id[i]]);
        }
        tmp_kws.reserve(100);
        while (index < result_str.size())
        {
            match_flag = false;
            for (int i = 0; i < m_pinyin_kws_list.size(); i++)
            {
                if (result_str[index] == m_pinyin_kws_list[i][0] && (index + m_pinyin_kws_list[i].size()) <= m_result_id.size())
                {
                    tmp_kws.clear();
                    for (int t = index; t < index + m_pinyin_kws_list[i].size(); t++)
                    {
                        tmp_kws.push_back(result_str[t]);
                    }
                    dist = get_edit_dist(tmp_kws, m_pinyin_kws_list[i]);
                    if (dist == 0 || (dist < 2 && m_pinyin_kws_list[i].size() > 3) || (dist < 3 && m_pinyin_kws_list[i].size() > 6))
                    {
                        match_flag = true;
                        out_index = i;
                        match_id.push_back(i);
                        break;
                    }
                }
            }
            if (match_flag)
            {
                index = index + m_pinyin_kws_list[out_index].size() - 1;
            }
            else
            {
                index++;
            }
        }
        if (match_id.size())
        {
            for (int i = 0; i < match_id.size(); i++)
            {
                out->append(m_hanzi_kws_list[match_id[i]]);
                out->append(" ");
            }
        }
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