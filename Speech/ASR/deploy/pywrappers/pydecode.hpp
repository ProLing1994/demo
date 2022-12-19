#ifndef _ASR_PYDECODE_H_
#define _ASR_PYDECODE_H_

#include "../common/decode.hpp"

#ifdef __cplusplus
extern "C" {
#endif

namespace ASR
{
	void* Decode_create();
	void Decode_delete(void* decode);
    void Decode_ctc_decoder(void* decode, float *input, int rows=35, int cols=480);
	int Decode_result_id_length(void* decode);
	void Decode_copy_result_id_to(void* decode, int *data);
} // namespace ASR

#ifdef __cplusplus
}
#endif

#endif // _ASR_PYDECODE_H_