#ifndef _ASR_COMMON_H_
#define _ASR_COMMON_H_

#include <string>
#include <vector>

namespace ASR
{
    bool ListDirectory(const char* directory, std::vector<std::string>& folders, std::vector<std::string>& filenames);
    
    std::vector<std::string> StringSplit(const std::string s, const std::string seperator);
} // namespace ASR
#endif // _ASR_COMMON_H_