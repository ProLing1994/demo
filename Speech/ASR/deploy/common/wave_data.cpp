#include <stdlib.h>
#include <string>
#include <string.h>

#include "wave_data.hpp"

namespace ASR
{
    Wave_Data::Wave_Data()
    {

    }

    Wave_Data::~Wave_Data()
    {
        clear_state();
    }

    int Wave_Data::load_data_length(const char *filename, int *length)
    {
        Wave_File_Head_S wav_head;
        FILE *fp;
        fp = fopen(filename, "rb");
        if (fp == nullptr)
        {
            printf("[ERROR:] Read wav failed!!\n");
            return -1;
        }
        fread(&wav_head, sizeof(struct Wave_File_Head_S), 1, fp); //read wavehead

        if(wav_head.channel != 1 || wav_head.sample_rate != 16000 || wav_head.bit_per_sample != 16 )  {
            printf("[ERROR:] Read wav failed!!, only support channel == %d/1, sample_rate == %d/16000, bit_per_sample == %d/16!!\n", \
                wav_head.channel, wav_head.sample_rate, wav_head.bit_per_sample);
            return -1;
        }
        // replace wav_head.data_size with men_len, becase some mistake is unkown in wav_head
        int32_t head_file_size = 36;  // 44 bytes(total head file size) - 8 bytes
        int32_t data_length = (wav_head.file_size - head_file_size) / sizeof(int16_t);
        int32_t memory_size = data_length * sizeof(int16_t);

        *length = static_cast<int>(data_length);
        return 0;
    }

    int Wave_Data::load_data(const char *filename)
    {
        Wave_File_Head_S wav_head;
        FILE *fp;
        fp = fopen(filename, "rb");
        if (fp == nullptr)
        {
            printf("[ERROR:] Read wav failed!!\n");
            return -1;
        }
        fread(&wav_head, sizeof(struct Wave_File_Head_S), 1, fp); //read wavehead

        if(wav_head.channel != 1 || wav_head.sample_rate != 16000 || wav_head.bit_per_sample != 16 )  {
            printf("[ERROR:] Read wav failed!!, only support channel == %d/1, sample_rate == %d/16000, bit_per_sample == %d/16!!\n", \
                wav_head.channel, wav_head.sample_rate, wav_head.bit_per_sample);
            return -1;
        }
        // replace wav_head.data_size with men_len, becase some mistake is unkown in wav_head
        int32_t head_file_size = 36;  // 44 bytes(total head file size) - 8 bytes
        int32_t data_length = (wav_head.file_size - head_file_size) / sizeof(int16_t);
        int32_t memory_size = data_length * sizeof(int16_t);

        if (m_wave_data.data == nullptr)
        {
            m_wave_data.data = (int16_t *)malloc(memory_size);
        }
        else
        {
            free(m_wave_data.data);
            m_wave_data.data = (int16_t *)malloc(memory_size);
        }

        m_wave_data.data_length = static_cast<unsigned int>(data_length);
        m_wave_data.fs = static_cast<unsigned short>(wav_head.sample_rate);
        memcpy(m_wave_data.data, wav_head.wave_data, memory_size);
        return 0;
    }

    void Wave_Data::copy_data_to(int16_t *data)
    {
        memcpy(data, m_wave_data.data, m_wave_data.data_length * sizeof(int16_t));
    }

    void Wave_Data::clear_state()
    {
        if(m_wave_data.data != nullptr)
        {
            free(m_wave_data.data);
        }
        m_wave_data.data_length = 0;
        m_wave_data.fs = 0;
    }

} // namespace ASR