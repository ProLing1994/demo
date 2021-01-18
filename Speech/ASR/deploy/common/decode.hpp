#ifndef _ASR_DECODE_H_
#define _ASR_DECODE_H_

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

namespace ASR
{
    class Decode
    {
    public:
        Decode();
        ~Decode();

    inline int result_id_length() const { return m_result_id.size(); }

    public:
        void ctc_decoder(float *input, int rows=35, int cols=480, bool greedy=true);
        void ctc_decoder(cv::Mat input, bool greedy = true);
        void copy_result_id_to(int *data);

    private:
        std::vector<int> m_result_id;

        int m_blank_id = 0;
    };
} // namespace ASR

#endif // _ASR_DECODE_H_