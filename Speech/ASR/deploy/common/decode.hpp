#ifndef _ASR_DECODE_H_
#define _ASR_DECODE_H_

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

namespace ASR
{
    class Decode
    {
    public:
        Decode();
        Decode(const std::vector<std::string> &keywords, 
                const std::vector<std::string> &symbol_list);
        Decode(const std::vector<std::string> &symbol_list, 
                const std::vector<std::string> &hanzi_kws_list, 
                const std::vector<std::vector<std::string>> &pinyin_kws_list);
        ~Decode();

        inline int result_id_length() const { return m_result_id.size(); }
        inline std::vector<int> result_id() const { return m_result_id; }

    public:
        void ctc_decoder(float *input, int rows=35, int cols=480, bool greedy=true);
        void ctc_decoder(cv::Mat input, bool greedy = true);

        void match_keywords_robust(std::string *output);
        int get_edit_dist(std::vector<std::string> string1, std::vector<std::string> string2);

        void show_symbol();
        void output_symbol(std::string *output);
        void copy_result_id_to(int *data);

    private:
        std::vector<int> m_result_id;

        int m_blank_id = 0;                         // 静音 SIL id, default：0
        
        std::vector<std::string> m_symbol_list;     // 符号
        std::vector<std::string> m_hanzi_kws_list;  // 汉字
        std::vector<std::vector<std::string>> m_pinyin_kws_list;    // 拼音
        std::vector<std::string> m_keywords;        // 关键词
    };
} // namespace ASR

#endif // _ASR_DECODE_H_