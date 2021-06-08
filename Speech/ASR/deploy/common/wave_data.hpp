#ifndef _ASR_WAVE_DATA_H_
#define _ASR_WAVE_DATA_H_

#include <vector>

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int int16_t;
typedef unsigned short int uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;

namespace ASR
{
	struct Wave_File_Head_S
	{
		char riff_char[4];		  //"RIFF"
		int32_t file_size;		  //file size - 8 bytes
		char wave_fmt_char[8];	  //"WAVE" and "fmt "
		int32_t format_length;	  //format length
		int16_t format_tag;		  //format tag
		int16_t channel;		  //channel
		int32_t sample_rate;	  //sample rate
		int32_t avg_bytes_sec; 	  //sample_rate*bit_per_sample*channel/8
		int16_t block_align;	  //block align
		int16_t bit_per_sample;	  //bit per sample
		char temp[2];			  //某些数据头存在差异
		char data_char[4]; 		  //"data"
		int32_t data_size;        //data size
		int16_t wave_data[3000000];
	};

	struct Wave_Data_S
	{
		Wave_Data_S(): 
			data(nullptr), 	
			data_length(0),
			fs(0),
			channel(1) {}

		int16_t *data;
		unsigned int data_length;
		unsigned short fs;
		unsigned int channel;
	};
	
	class Wave_Data
	{
	public:
		Wave_Data();
		~Wave_Data();

		inline int data_length() const { return m_wave_data.data_length; } 
		inline int fs() const { return m_wave_data.fs; } 
		inline int16_t *data() { return m_wave_data.data; }
		inline int channel() { return m_wave_data.channel; }
	
	public:
		int load_data(const char *filename);
		void write_data(int16_t *data, int data_length, int fs, const char *filename);
		void copy_data_to(int16_t *data);
		void clear_state();
		short get(int32_t i, int8_t c=0);
		
	private:
		Wave_Data_S m_wave_data;
	};

	int load_pcm_data(const char *filename, std::vector<short> &wave_data);
} // namespace ASR

#endif // _ASR_WAVE_DATA_H_