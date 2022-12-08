#include <iostream>

#include "file_system.h"

#ifdef _WIN32
#include "common/3rd_party/dirent.h"
#else
#include "dirent.h"
#endif

namespace yh_common {

bool list_directory(const char* directory, std::vector<std::string>& folders, std::vector<std::string>& filenames)
{
  DIR *dir;
  struct dirent *ent;

  /* Open directory stream */
  dir = opendir (directory);
  if (dir != NULL) {

    /* Get all files and directories within the directory */
    while ((ent = readdir (dir)) != NULL) {
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


}
