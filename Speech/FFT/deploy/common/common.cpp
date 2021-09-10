#include <iostream>
#include <ctime>

#include "dirent.h"

#include "common.hpp"

#ifdef _ANDROID
#include <android/log.h>
#endif

namespace ASR{

	bool ListDirectory(const char* directory, std::vector<std::string>& folders, std::vector<std::string>& filenames)
	{
		DIR *dir;
		struct dirent *ent;

		/* Open directory stream */
		dir = opendir(directory);
		if (dir != NULL) {

			/* Get all files and directories within the directory */
			while ((ent = readdir(dir)) != NULL) {
				switch (ent->d_type) {
				case DT_REG:
					filenames.push_back(ent->d_name);
					break;

				case DT_DIR:
					folders.push_back(ent->d_name);
					break;

				default:
					// ignore
					break;
				}
			}

			closedir (dir);
			return true;

		} else {
			/* Could not open directory */
			return false;
		}
	}

	bool ListDirectorySuffix(const char* directory, std::vector<std::string>& filenames, std::string suffix)
	{
		DIR *dir;
		struct dirent *ent;

		/* Open directory stream */
		dir = opendir(directory);
		if (dir != NULL) {

			/* Get all files and directories within the directory */
			while ((ent = readdir(dir)) != NULL) {
			
				std::string filename= ent->d_name;
				if (filename.find(suffix) != filename.npos){
					filenames.push_back(ent->d_name);
					continue;
				}
			}

			closedir (dir);
			return true;

		} else {
			/* Could not open directory */
			return false;
		}
	}


	std::vector<std::string> StringSplit(const std::string &s, const std::string &seperator)
	{	
		std::vector<std::string> result;
		typedef std::string::size_type string_size;

		int swidth = seperator.size();
		string_size i = 0, j = 0;

		while ( (j + swidth) <= s.size() )
		{	
			if ( s.substr(j, swidth) == seperator )
			{   
				if ( j-i > 0 )
				{
					result.push_back( s.substr(i,(j-i)) );
				}
				i = j + swidth;
				j = i;
			}
			else
			{   
				j ++;
			}
		}

		if ( i < s.size() )
		{   
			result.push_back( s.substr(i, s.size()-i) );
		}
		
		return result;
	}

	// 下面是Logger相关

	Logger::Logger(): m_level(0), 
				      m_logger_name("core"),
					  m_stacked_message("")
	{
		#ifdef _ANDROID
		m_LOG_TAG.append("[ASR.");
		m_LOG_TAG.append(m_logger_name);
		m_LOG_TAG.append("]");
		#endif
	}

	Logger::Logger( int level ): m_level(level), 
								 m_logger_name("core"),
								 m_stacked_message("")
	{
		#ifdef _ANDROID
		m_LOG_TAG.append("[ASR.");
		m_LOG_TAG.append(m_logger_name);
		m_LOG_TAG.append("]");
		#endif		
	}

	Logger::Logger( std::string name, int level ): m_level(level), 
												   m_logger_name(name),
												   m_stacked_message("")
	{
		#ifdef _ANDROID
		m_LOG_TAG.append("[ASR.");
		m_LOG_TAG.append(m_logger_name);
		m_LOG_TAG.append("]");
		#endif	
	}

	void Logger::debug( const std::string &s )
	{
		if ( m_level == 0 )
		{
			char timestamp[32];
			time_t timep;
			time (&timep);
    		strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&timep) );
			#ifdef _ANDROID
			char messsage[128];
			sprintf(messsage, "[%s][DEBUG:] %s", timestamp, s.c_str());
			__android_log_write(ANDROID_LOG_INFO, m_LOG_TAG.c_str(), messsage);
			#else
			printf( "[%s][ASR.%s][DEBUG:] %s \n", timestamp, m_logger_name.c_str(), s.c_str() );
			#endif
		}
	}

	void Logger::debug_s()
	{
		debug( get_stacked_s() );
	}

	void Logger::info( const std::string &s )
	{
		char timestamp[32];
		time_t timep;
		time (&timep);
		strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&timep) );
		#ifdef _ANDROID
		char messsage[128];
		sprintf(messsage, "[%s][INFO:] %s", timestamp, s.c_str());
		__android_log_write(ANDROID_LOG_INFO, m_LOG_TAG.c_str(), messsage);
		#else
		printf( "[%s][ASR.%s][INFO:] %s \n", timestamp, m_logger_name.c_str(), s.c_str() );
		#endif
	}

	void Logger::info_s()
	{
		info( get_stacked_s() );
	}

	void Logger::error(const std::string &s )
	{
		char timestamp[32];
		time_t timep;
		time (&timep);
		strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&timep) );
		#ifdef _ANDROID
		char messsage[128];
		sprintf(messsage, "[%s][ERROR:] %s", timestamp, s.c_str());
		__android_log_write(ANDROID_LOG_INFO, m_LOG_TAG.c_str(), messsage);
		#else
		printf( "[%s][ASR.%s][ERROR:] %s \n", timestamp, m_logger_name.c_str(), s.c_str() );
		#endif
	}	

	void Logger::error_s()
	{
		error( get_stacked_s() );
	}

	std::string Logger::get_stacked_s()
	{
		std::string temp_s = m_stacked_message;
		m_stacked_message.clear();
		return temp_s;
	}

	Logger & Logger::operator<<( const std::string & item )
	{
		m_stacked_message += item;
		return *this;
	}

	Logger & Logger::operator<<( const int & item )
	{
		m_stacked_message += std::to_string(item);
		return *this;
	}

	Logger & Logger::operator<<( const double & item )
	{
		m_stacked_message += std::to_string(item);
		return *this;
	}	

	// Logger & Logger::operator<<( const bool & item )
	// {	
	// 	if (item)
	// 	{ m_stacked_message += "True"; }
	// 	else
	// 	{ m_stacked_message += "False"; }
	// 	return *this;
	// }	

	Logger getLogger(std::string name)
	{
		#ifdef _DEBUG
		Logger logger(name, 0);
		#else
		Logger logger(name, 1);
		#endif

		return logger;
	}

} //namespace ASR