#ifndef _ASR_WAV_LOADER_H_
#define _ASR_WAV_LOADER_H_

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int int16_t;
typedef unsigned short int uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;

#ifdef __cplusplus
extern "C" {
#endif

namespace ASR
{
	struct Wave_File_Head_S
	{
		char riff_char[4];		  //"RIFF"
		int32_t file_size;		  //file size - 8 bytes
		char wave_fmt_char[8];	  //"wave" and "fmt"
		int32_t format_length;	  //format length
		int16_t format_tag;		  //format tag
		int16_t channel;		  //channel
		int32_t sample_rate;	  //sample rate
		int32_t avg_bytes_sec; 	  //sample_rate*bit_per_sample*channel/8
		int16_t block_align;	  //block align
		int16_t bit_per_sample;	  //bit per sample
		char data_char[4]; 		  //"data"
		int32_t data_size;        //data size
		int16_t wave_data[3000000];
	};

	struct Wave_Data_S
	{
		int16_t *data = NULL;
		unsigned int data_length = 0;
		unsigned short fs = 0;
	};
	
	unsigned int ReadWaveLength(const char *filename);
    void ReadWave(const char *filename, Wave_Data_S *wave_data);
	void Wave_copy_data_to(Wave_Data_S *wave_data, int16_t *data);
} // namespace ASR

#ifdef __cplusplus
}
#endif

#endif // _ASR_WAV_LOADER_H_