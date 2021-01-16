#include "pywave_data.hpp"

namespace ASR
{
    void* Wave_Data_create()
    {
        return new Wave_Data();
    }

    void Wave_Data_delete(void* wave_data)
    {
        Wave_Data* obj = static_cast<Wave_Data*>(wave_data);
        delete obj;
    }

    int Wave_Data_load_data(void* wave_data, const char *filename)
    {
        Wave_Data* obj = static_cast<Wave_Data*>(wave_data);
        return obj->load_data(filename);
    }

    int Wave_Data_length(void* wave_data)
    {
        Wave_Data* obj = static_cast<Wave_Data*>(wave_data);
        return obj->data_length();
    }

    void Wave_Data_copy_data_to(void* wave_data, int16_t *data)
    {
        Wave_Data* obj = static_cast<Wave_Data*>(wave_data);
        obj->copy_data_to(data);
    }
    
    void Wave_Data_clear_state(void* wave_data)
    {
        Wave_Data* obj = static_cast<Wave_Data*>(wave_data);
        obj->clear_state();
    }
} // namespace ASR