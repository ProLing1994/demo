#include "pydecode.hpp"

namespace ASR
{
    void* Decode_create()
    {
        return new Decode();
    }

    void Decode_delete(void* decode)
    {
        Decode* obj = static_cast<Decode*>(decode);
        delete obj;
    }

    void Decode_ctc_decoder(void* decode, float *input, int rows, int cols)
    {
        Decode* obj = static_cast<Decode*>(decode);
        obj->ctc_decoder(input, rows, cols);
    }

    int Decode_length(void* decode)
    {
        Decode* obj = static_cast<Decode*>(decode);
        return obj->result_id_length();
    }

    void Decode_copy_result_id_to(void* decode, int *data)
    {
        Decode* obj = static_cast<Decode*>(decode);
        obj->copy_result_id_to(data);
    }
} // namespace ASR