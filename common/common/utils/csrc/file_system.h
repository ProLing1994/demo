#ifndef MD_UTILITY_FILE_SYSTEM_H
#define MD_UTILITY_FILE_SYSTEM_H

#include <vector>
#include <string>

namespace yh_common {

/*! \brief list the content of a directory
 *
 *  \param directory       the directory to query
 *  \param folders         the folder names under the directory
 *  \param filenames       the file names under the directory
 */
bool list_directory(const char* directory,
                    std::vector<std::string>& folders, std::vector<std::string>& filenames);




}

#endif
