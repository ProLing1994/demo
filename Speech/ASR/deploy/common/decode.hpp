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
        Decode(const std::vector<std::string> &keywords, const std::vector<std::string> &symbol_list);
        Decode(const int blank_id, const std::vector<std::string> &symbol_list, 
                const std::vector<std::string> &hanzi_kws_list, const std::vector<std::vector<std::string>> &pinyin_kws_list);
        ~Decode();

    inline int result_id_length() const { return m_result_id.size(); }
    inline std::vector<int> result_id() const { return m_result_id; }

    public:
        void ctc_decoder(float *input, int rows=35, int cols=480, bool greedy=true);
        void ctc_decoder(cv::Mat input, bool greedy = true);

        void copy_result_id_to(int *data);

    private:
        void match_keywords_robust(std::string *out);
        int get_edit_dist(std::vector<std::string> string1, std::vector<std::string> string2);

    private:
        std::vector<int> m_result_id;

        int m_blank_id = 0;
        
        std::vector<std::string> m_symbol_list;
        std::vector<std::string> m_keywords;
        std::vector<std::string> m_hanzi_kws_list;
        std::vector<std::vector<std::string>> m_pinyin_kws_list;
    };
} // namespace ASR

#endif // _ASR_DECODE_H_