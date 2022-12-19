#include <algorithm>

#include "common.hpp"
#include "decode.hpp"

namespace ASR
{
    Decode::Decode()
    {
    }

    Decode::Decode(const std::vector<std::string> &keywords, const std::vector<std::string> &symbol_list)
    {
        m_keywords.reserve(keywords.size());
        for(unsigned int i = 0; i < keywords.size(); i++)
        {
            m_keywords.push_back(keywords[i]);
        }

        m_symbol_list.reserve(symbol_list.size());
        for(unsigned int i = 0; i < symbol_list.size(); i++)
        {
            m_symbol_list.push_back(symbol_list[i]);
        }
    }

    Decode::Decode(const std::vector<std::string> &symbol_list, 
                    const std::vector<std::string> &hanzi_kws_list, 
                    const std::vector<std::vector<std::string>> &pinyin_kws_list,
                    const std::vector<std::vector<std::string>> &english_kws_list)
    {
        m_symbol_list.reserve(symbol_list.size());
        for(unsigned int i = 0; i < symbol_list.size(); i++)
        {
            m_symbol_list.push_back(symbol_list[i]);
        }

        m_hanzi_kws_list.reserve(hanzi_kws_list.size());
        for(unsigned int i = 0; i < hanzi_kws_list.size(); i++)
        {
            m_hanzi_kws_list.push_back(hanzi_kws_list[i]);
        }

        m_pinyin_kws_list.reserve(pinyin_kws_list.size());
        for(unsigned int i = 0; i < pinyin_kws_list.size(); i++)
        {
            m_pinyin_kws_list.push_back(pinyin_kws_list[i]);
        }
        
        m_english_kws_list.reserve(english_kws_list.size());
        for(unsigned int i = 0; i < english_kws_list.size(); i++)
        {
            m_english_kws_list.push_back(english_kws_list[i]);
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
        // clear 
        m_result_id.clear();

        int frame_num = input.rows;
        int feature_num = input.cols;

        int last_max_id = 0;
        if (greedy)
        {
            float max_value = 0;
            int max_id = 0;
            for (int i = 0; i < frame_num; i++)
            {
                max_value = input.at<float>(i, 0);
                max_id = 0;
                for (int j = 1; j < feature_num; j++)
                {
                    if (input.at<float>(i, j) > max_value)
                    {
                        max_value = input.at<float>(i, j);
                        max_id = j;
                    }
                }
                if (max_id != (m_blank_id) && last_max_id != max_id)
                {
                    m_result_id.push_back(max_id);
                    last_max_id = max_id;

                    // // debug
                    // std::cout << "max_id: " << max_id << std::endl;
                    // for (int j = 1; j < feature_num; j++)
                    // {   
                    //     std::cout << i << ", " << j << ": " << input.at<float>(i, j) << std::endl;
                    // }
                }
            }
        }

        return;
    }

    void Decode::show_result_id()
    {
        for(unsigned int i = 0; i < m_result_id.size(); i++)
        {
            std::cout << m_result_id[i] <<" ";
        }
        std::cout << std::endl;
    }

    void Decode::show_symbol()
    {
        for(unsigned int i = 0; i < m_result_id.size(); i++)
        {
            std::cout << m_symbol_list[m_result_id[i]] << " ";
        }
        std::cout << std::endl;
    }

    void Decode::show_symbol_english()
    {
        std::string symbol;
        for(unsigned int i = 0; i < m_result_id.size(); i++)
        {
            symbol = m_symbol_list[m_result_id[i]];
            if(symbol[0] == '_')
            {
                if(i != 0)
                    std::cout << " ";
                std::cout << symbol.substr(1);
            }
            else
            {
                std::cout << symbol;
            }
        }
        std::cout << std::endl;
    }

    void Decode::output_symbol(std::string *output)
    {
        for(unsigned int i = 0; i < m_result_id.size(); i++)
        {
            output->append(m_symbol_list[m_result_id[i]]);
            output->append(" ");
        }
    }

    void Decode::output_symbol_english(std::string *output)
    {
        std::string symbol;
        for(unsigned int i = 0; i < m_result_id.size(); i++)
        {
            symbol = m_symbol_list[m_result_id[i]];
            if(symbol[0] == '_')
            {
                if(i != 0)
                    output->append(" ");
                output->append(symbol.substr(1));
            }
            else
            {
                output->append(symbol);
            }
        }
        if(m_result_id.size())
            output->append(" ");
    }

    void Decode::match_keywords_robust(std::string *output)
    {
        unsigned int index = 0;
        int dist = 0;
        int out_index = 10000;
        bool match_flag = false;
        std::vector<int> match_id;
        std::vector<std::string> tmp_kws;
        std::vector<std::string> result_str;

        for (unsigned int i = 0; i < m_result_id.size(); i++)
        {
            result_str.push_back(m_symbol_list[m_result_id[i]]);
        }
        
        tmp_kws.reserve(100);
        while (index < result_str.size())
        {
            match_flag = false;
            for (unsigned int i = 0; i < m_pinyin_kws_list.size(); i++)
            {   
                // for(int j = 0; j < m_pinyin_kws_list[i].size(); j++)
                // {
                //     std::cout << m_pinyin_kws_list[i][j] << " ";
                // }
                // std::cout << std::endl;

                if (result_str[index] == m_pinyin_kws_list[i][0] && (index + m_pinyin_kws_list[i].size()) <= m_result_id.size())
                {
                    tmp_kws.clear();
                    for (unsigned int t = index; t < index + m_pinyin_kws_list[i].size(); t++)
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
            for (unsigned int i = 0; i < match_id.size(); i++)
            {
                output->append(m_hanzi_kws_list[match_id[i]]);
                output->append(" ");
            }
        }
    }

    void Decode::match_keywords_english(std::string *output)
    {
        // init 
        std::string english_symbol;
        std::vector<std::string> english_symbol_vector;
        std::vector<std::string> match_stack_vector;

        // get symbol
        output_symbol_english(&english_symbol);
        english_symbol_vector = ASR::StringSplit(english_symbol, " ");
        
        // 遍历字符串
        for (unsigned int i = 0; i < english_symbol_vector.size(); i++) {
            // 遍历模板
            bool find_kws_bool = false;
            for (unsigned int j = 0; j < m_english_kws_list.size(); j++) {
                // 遍历模板成员
                for (unsigned int k = 0; k < m_english_kws_list[j].size(); k++) {
                    // To do：匹配策略，即使更新
                    // std::cout << english_symbol_vector[i] << ", " << m_english_kws_list[j][k] << std::endl;
                    if (match_english_string(m_english_kws_list[j][k], english_symbol_vector[i])) {
                        // 若遍历至模板成员最后一个成员，遍历堆中结果，判断模板成员是否均存在
                        if(k == m_english_kws_list[j].size() - 1) {
                            if(match_stack_vector.size() >= m_english_kws_list[j].size() - 1) {
                                // 针对长度为 1 的匹配词
                                // std::cout << m_english_kws_list[j].size() - 1 << std::endl;
                                if(m_english_kws_list[j].size() - 1 == 0)
                                    find_kws_bool = true;

                                // 针对长度大于 1 的匹配词，逆序匹配
                                for (unsigned int m = 0; m < m_english_kws_list[j].size() - 1; m++) {
                                    // std::cout << match_stack_vector[match_stack_vector.size()-1 - m] << std::endl;
                                    // std::cout << m_english_kws_list[j][m_english_kws_list[j].size() -2 - m] << std::endl;
                                    if(match_stack_vector[match_stack_vector.size()-1 - m] == \
                                            m_english_kws_list[j][m_english_kws_list[j].size() -2 - m])
                                        find_kws_bool = true;
                                    else
                                        find_kws_bool = false;
                                }
                            }
                        }
                        else {
                            // 建堆 push
                            match_stack_vector.push_back(m_english_kws_list[j][k]);
                        }
                        
                        if(find_kws_bool) {
                            // 更新输出
                            for (unsigned int i = 0; i < m_english_kws_list[j].size(); i++) {
                                output->append(m_english_kws_list[j][i]);
                                output->append(" ");
                            }

                            // 弹堆 ppo
                            for (unsigned int m = 0; m < m_english_kws_list[j].size() - 1; m++) {
                                match_stack_vector.pop_back();
                            }
                        }
                    }
                }

                // 若找到模板，则遍历下一个字符串
                if(find_kws_bool)
                    break;
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
        for(unsigned int i = 0; i < m_result_id.size(); i++)
        {
            temp_data[i] = m_result_id[i];
        }
        memcpy(data, temp_data, m_result_id.size() * sizeof(int));
    }

    bool Decode::match_english_string(const std::string output1, const std::string output2)
    {
        // 匹配策略，任意匹配 [ing, ed, s]
        std::string output2_substr;
        output2_substr = output2.substr(0, output1.size());
        if(output2_substr.find(output1) != output2_substr.npos) {
            return true;
        }
        else
            return false;
    }
} // namespace ASR