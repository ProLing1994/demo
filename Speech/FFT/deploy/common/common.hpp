#ifndef _ASR_COMMON_H_
#define _ASR_COMMON_H_

#include <string>
#include <vector>

namespace ASR
{
    bool ListDirectory(const char* directory, std::vector<std::string>& folders, std::vector<std::string>& filenames);
    bool ListDirectorySuffix(const char* directory, std::vector<std::string>& filenames, std::string suffix=".wav");
    
    std::vector<std::string> StringSplit(const std::string &s, const std::string &seperator);

    class Logger
    {
        public:
            Logger();
            Logger( int level );
            Logger( std::string name, int level );
            ~Logger() {}
        
        public:
            void info(const std::string &s);
            void info_s();
            void debug(const std::string &s);
            void debug_s();
            void error(const std::string &s);
            void error_s();

            std::string get_stacked_s();

            Logger & operator<<( const std::string & item );
            Logger & operator<<( const int & item );
            Logger & operator<<( const double & item );
            // Logger & operator<<( const bool & item );

        private:
            int m_level;
            std::string m_logger_name;
            std::string m_stacked_message;

            #ifdef _ANDROID
            std::string m_LOG_TAG;
            #endif
    };

    Logger getLogger(std::string name="core");

} // namespace ASR
#endif // _ASR_COMMON_H_