#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>

#include "wave_data.hpp"


namespace ASR
{
    Wave_Data::Wave_Data()
    {
        m_wave_data = Wave_Data_S();
    }

    Wave_Data::~Wave_Data()
    {
        clear_state();
    }


    int Wave_Data::load_data(const char *filename)
    {
        Wave_File_Head_S wav_head;
        FILE *fp;
        fp = fopen(filename, "rb");
        if (fp == nullptr)
        {
            printf("[ERROR:] %s, %d: Read wav failed.\n", __FUNCTION__, __LINE__);
            return -1;
        }
        fread(&wav_head, sizeof(struct Wave_File_Head_S), 1, fp); //read wavehead

        // check
        if(wav_head.channel != 1 || wav_head.sample_rate != 16000 || wav_head.bit_per_sample != 16 )  
        {
            printf("[ERROR:] %s, %d: Read wav failed!!, only support channel == %d/1, sample_rate == %d/16000, bit_per_sample == %d/16.\n", \
                    __FUNCTION__, __LINE__, wav_head.channel, wav_head.sample_rate, wav_head.bit_per_sample);
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

        // std::cout << "wav_head.riff_char: " << wav_head.riff_char << ", wav_head.file_size: " << wav_head.file_size \
        //             << ", wav_head.wave_fmt_char: " << wav_head.wave_fmt_char \
        //             << ", wav_head.format_length: " << wav_head.format_length \
        //             << ", wav_head.format_tag: " << wav_head.format_tag \
        //             << ", wav_head.channel: " << wav_head.channel \
        //             << ", wav_head.sample_rate: " << wav_head.sample_rate \
        //             << ", wav_head.avg_bytes_sec: " << wav_head.avg_bytes_sec \
        //             << ", wav_head.block_align: " << wav_head.block_align \
        //             << ", wav_head.bit_per_sample: " << wav_head.bit_per_sample \
        //             << ", wav_head.data_char: " << wav_head.data_char \
        //             << ", wav_head.data_size: " << wav_head.data_size << std::endl;
        return 0;
    }

    void Wave_Data::write_data(int16_t *data, unsigned int data_length, unsigned short fs, const char *filename)
    {
        FILE *fp;
        fp = fopen(filename, "wb");
        fwrite("RIFF", sizeof(char), 4, fp);

        int32_t head_file_size = 36;
        int32_t file_length = head_file_size + data_length * sizeof(int16_t);
        fwrite(&file_length, sizeof(int32_t) , 1, fp);

        fwrite("WAVE", sizeof(char), 4, fp);
        fwrite("fmt ", sizeof(char), 4, fp);

        int32_t format_length = 18;
        fwrite(&format_length, sizeof(int32_t), 1, fp);

		int16_t format_tag = 1;		  
        fwrite(&format_tag, sizeof(int16_t), 1, fp);

		int16_t channel = 1;
        fwrite(&channel, sizeof(int16_t), 1, fp);

		int32_t sample_rate = fs;
        fwrite(&sample_rate, sizeof(int32_t), 1, fp);

        int16_t bit_per_sample = 16;
		int32_t avg_bytes_sec = channel * sample_rate * bit_per_sample / 8;
        fwrite(&avg_bytes_sec, sizeof(int32_t), 1, fp);

		int16_t block_align = channel * bit_per_sample / 8;
        fwrite(&block_align, sizeof(int16_t), 1, fp);

        fwrite(&bit_per_sample, sizeof(int16_t), 1, fp);
        fwrite("  data", sizeof(char), 6, fp);

		int32_t data_size = data_length * sizeof(int16_t);
        fwrite(&data_size, sizeof(int32_t), 1, fp);

        fwrite(data, sizeof(int16_t), data_length, fp);
        fclose(fp);
        return;
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