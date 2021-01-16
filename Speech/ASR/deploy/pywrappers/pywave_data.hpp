#ifndef _ASR_PYWAVE_DATA_H_
#define _ASR_PYWAVE_DATA_H_

#include "../common/wave_data.hpp"

#ifdef __cplusplus
extern "C" {
#endif

namespace ASR
{
	void* Wave_Data_create();
	void Wave_Data_delete(void* wave_data);
	int Wave_Data_load_data(void* wave_data, const char *filename);
	int Wave_Data_length(void* wave_data);
	void Wave_Data_copy_data_to(void* wave_data, int16_t *data);
	void Wave_Data_clear_state(void* wave_data);
} // namespace ASR

#ifdef __cplusplus
}
#endif

#endif // _ASR_PYWAVE_DATA_H_