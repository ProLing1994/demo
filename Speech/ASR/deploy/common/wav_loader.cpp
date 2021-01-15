#include <stdlib.h>
#include <string>
#include <string.h>

#include "wav_loader.hpp"

namespace ASR
{
    unsigned int ReadWaveLength(const char *filename)
    {
        ASR::Wave_File_Head_S wav_head;
        FILE *fp;
        fp = fopen(filename, "rb");
        if (fp == NULL)
        {
            printf("[ERROR:] Read wav failed!!\n");
        }
        fread(&wav_head, sizeof(struct ASR::Wave_File_Head_S), 1, fp); //read wavehead

        if(wav_head.channel != 1 || wav_head.sample_rate != 16000 || wav_head.bit_per_sample != 16 )  {
            printf("[ERROR:] Read wav failed!!, only support channel == %d/1, sample_rate == %d/16000, bit_per_sample == %d/16!!\n", \
                wav_head.channel, wav_head.sample_rate, wav_head.bit_per_sample);
            return -1;
        }
        // replace wav_head.data_size with men_len, becase some mistake is unkown in wav_head
        int32_t head_file_size = 36;  // 44 bytes(total head file size) - 8 bytes
        int32_t data_length = (wav_head.file_size - head_file_size) / sizeof(int16_t);
        int32_t memory_size = data_length * sizeof(int16_t);

        return static_cast<unsigned int>(data_length);
    }

    void ReadWave(const char *filename, Wave_Data_S *wave_data)
    {
        ASR::Wave_File_Head_S wav_head;
        FILE *fp;
        fp = fopen(filename, "rb");
        if (fp == NULL)
        {
            printf("[ERROR:] Read wav failed!!\n");
        }
        fread(&wav_head, sizeof(struct ASR::Wave_File_Head_S), 1, fp); //read wavehead

        if(wav_head.channel != 1 || wav_head.sample_rate != 16000 || wav_head.bit_per_sample != 16 )  {
            printf("[ERROR:] Read wav failed!!, only support channel == %d/1, sample_rate == %d/16000, bit_per_sample == %d/16!!\n", \
                wav_head.channel, wav_head.sample_rate, wav_head.bit_per_sample);
            return;
        }
        // replace wav_head.data_size with men_len, becase some mistake is unkown in wav_head
        int32_t head_file_size = 36;  // 44 bytes(total head file size) - 8 bytes
        int32_t data_length = (wav_head.file_size - head_file_size) / sizeof(int16_t);
        int32_t memory_size = data_length * sizeof(int16_t);

        if (wave_data->data == NULL)
        {
            wave_data->data = (int16_t *)malloc(memory_size);
        }
        else
        {
            free(wave_data->data);
            wave_data->data = (int16_t *)malloc(memory_size);
        }

        wave_data->data_length = static_cast<unsigned int>(data_length);
        wave_data->fs = static_cast<unsigned short>(wav_head.sample_rate);
        memcpy(wave_data->data, wav_head.wave_data, memory_size);
    }

    void Wave_copy_data_to(Wave_Data_S *wave_data, int16_t *data)
    {
        memcpy(data, wave_data->data, wave_data->data_length * sizeof(int16_t));
    }
} // namespace ASR