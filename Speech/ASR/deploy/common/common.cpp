#include "dirent.h"

#include "common.hpp"

bool ASR::ListDirectory(const char* directory, std::vector<std::string>& folders, std::vector<std::string>& filenames)
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

std::vector<std::string> ASR::StringSplit(const std::string s, const std::string seperator)
{
	std::vector<std::string> result;
	typedef std::string::size_type string_size;
	string_size i = 0;
	while (i != s.size())
	{
		int flag = 0;
		while (i != s.size() && flag == 0)
		{
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x])
				{
					++i;
					flag = 0;
					break;
				}
		}
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0)
		{
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x])
				{
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j)
		{
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}