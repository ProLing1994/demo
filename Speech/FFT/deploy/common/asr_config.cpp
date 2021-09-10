
#include <iostream>
#include <math.h>
#include <map>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "asr_config.h"

using namespace std;

inline int rmSprintf(char dst[], const char *fmt, ...)
{
	int cnt = -1;
	va_list args;
	va_start(args, fmt);

#ifdef WIN32
	cnt = vsprintf(dst, fmt, args);
#else
	memset(dst, 0, sizeof(dst));
	cnt = vsnprintf(dst, sizeof(dst), fmt, args);
#endif // WIN32
	va_end(args);

	return (cnt);
}

/*************************************************************************
 * Function:			rmCConfig
 * Descroption:	构造函�?
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 * Return:
 * Support:
 ANDROID:	yes
 HISI   :			yes
 WIN32  :		yes
 ************************************************************************/
rmCConfig::rmCConfig()
{
	memset(m_szKey, 0, sizeof(m_szKey));
}

rmCConfig::rmCConfig(const char *pathName)
{
	memset(m_szKey, 0, sizeof(m_szKey));

	OpenFile(pathName);
}

/*************************************************************************
 * Function:			~rmCConfig
 * Descroption:	析构函数
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 * Return:
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
rmCConfig::~rmCConfig()
{
	//WriteFile();
	m_Map.clear();
}

/*************************************************************************
 * Function:			FileSize
 * Descroption:	get file size, return file number of lines
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 filename:
 * Return:			 int
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
int rmCConfig::FileSize(const char *filename)
{
	std::ifstream input(const_cast<char *>(filename), ios::in | ios::binary);
	if (!input.is_open())
	{
		printf("%s open failed!\n", filename);
		return -1;
	}

	std::string line;
	int cnt(0);
	while (getline(input, line))
		if (line.length() > 0)
			cnt++;

	input.close();
	return cnt;
}

/*************************************************************************
 * Function:			CopyFile
 * Descroption:
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 srcFile:
 dstFile:
 * Return:			 bool
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
bool rmCConfig::CopyFile(const char *srcFile, const char *dstFile)
{
	std::ifstream input(const_cast<char *>(srcFile), ios::in | ios::binary);
	if (!input.is_open())
	{
		printf("%s open failed!\n", srcFile);
		return false;
	}
	std::ofstream output(const_cast<char *>(dstFile), ios::out | ios::binary);
	if (!output.is_open())
	{
		printf("%s open failed!\n", dstFile);
		return false;
	}
	std::string line;

	while (getline(input, line))
		output << line << "\n";

	input.close();
	output.close();

#ifndef WIN32
	system("sync");
#endif

	return true;
}

/*************************************************************************
 * Function:			OpenFile
 * Descroption:	打开文件函数
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 pathName:
 type:
 * Return:			 CONFIG_RES_E
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
CONFIG_RES_E rmCConfig::OpenFile(const char *pathName, bool isSync)
{
	string srcfileName = pathName;
	string bakfileName = srcfileName + ".bak";

	if (isSync)
	{
		if (!CopyFile(srcfileName.data(), bakfileName.data()))
		{
			printf("%s recover error!\n", bakfileName.data());
			return CONFIG_SYNC_ERROR;
		}
	}

	string szLine, szMainKey, szLastMainKey, szSubKey;
	KEYMAP mLastMap;
	int nIndexPos = -1;
	int nLeftPos = -1;
	int nRightPos = -1;
	ifstream m_fp(pathName, ios::in | ios::binary);

	if (!m_fp.is_open())
	{
		printf("open inifile %s error!\n", pathName);
		return CONFIG_OPENFILE_ERROR;
	}

	file_path = pathName;
	m_Map.clear();
	int count = 0;

	while (!m_fp.eof())
	{
		count++;
		getline(m_fp, szLine);

		//判断是否是主�?
		nLeftPos = szLine.find("[");
		nRightPos = szLine.find("]");
		if (nLeftPos != string::npos && nRightPos != string::npos)
		{
			if (nRightPos - nLeftPos <= 1)
				continue;

			szLastMainKey = szLine.substr(nLeftPos + 1, nRightPos - nLeftPos - 1);
			mLastMap.clear();
			m_Map[szLastMainKey] = mLastMap;
		}
		else
		{
			//是否是子�?
			if (nIndexPos = szLine.find("="), string::npos != nIndexPos)
			{
				string szSubKey, szSubValue;

				szSubKey = szLine.substr(0, nIndexPos);
				nLeftPos = szSubKey.find("<");
				nRightPos = szSubKey.find(">");
				if (nLeftPos == string::npos || nRightPos == string::npos)
				{
					continue;
				}
				szSubKey = szSubKey.substr(nLeftPos + 1, nRightPos - nLeftPos - 1);

				szSubValue = szLine.substr(nIndexPos + 1, szLine.length() - nIndexPos - 1);
				nLeftPos = szSubValue.find("<");
				nRightPos = szSubValue.find(">");
				if (nLeftPos == string::npos || nRightPos == string::npos)
				{
					continue;
				}
				szSubValue = szSubValue.substr(nLeftPos + 1, nRightPos - nLeftPos - 1);

				m_Map[szLastMainKey][szSubKey].value = szSubValue;
				m_Map[szLastMainKey][szSubKey].line_pos = count;
			}
			else
			{
				//TODO:不符合ini键值模板的内容 如注释等
			}
		}
	}

	//关闭文件
	if (m_fp.is_open())
	{
		m_fp.close();
	}

	on_change.clear();

	return CONFIG_SUCCESS;
}

void rmCConfig::KeySort(KEYVECTOR &v, int left, int right)
{
	if (left < right)
	{
		KeyValue key = v[left];
		int low = left;
		int high = right;
		while (low < high)
		{
			while (low < high && v[high].line_pos > key.line_pos)
				high--;
			v[low] = v[high];
			while (low < high && v[low].line_pos < key.line_pos)
				low++;
			v[high] = v[low];
		}
		v[low] = key;
		KeySort(v, left, low - 1);
		KeySort(v, low + 1, right);
	}
}

/*************************************************************************
 * Function:			WriteFile
 * Descroption:	写入配置文件
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 * Return:			 CONFIG_RES_E
 * Support:
 ANDROID:	yes
 HISI   :			yes
 WIN32  :		yes
 ************************************************************************/
CONFIG_RES_E rmCConfig::WriteFile()
{
	string szLine;
	vector<string> vLines;
	ifstream m_fp_in(const_cast<char *>(file_path.c_str()), ios::in | ios::binary);
	if (!m_fp_in.is_open())
	{
		printf("%s open error\n", file_path.c_str());
		return CONFIG_OPENFILE_ERROR;
	}

	int onchange_size = on_change.size();
	int onchange_pos(0), line_num(0);
	KeySort(on_change, 0, onchange_size - 1);

	while (!m_fp_in.eof())
	{
		line_num++;
		getline(m_fp_in, szLine);
		if (onchange_pos < onchange_size && line_num == on_change[onchange_pos].line_pos)
		{
			/*
			* 写文件，存入所需的内�?
			*/
			int start_pos = szLine.rfind("<") + 1;
			int end_pos = szLine.rfind(">");
			szLine.replace(start_pos, end_pos - start_pos, on_change[onchange_pos].value);
			onchange_pos++;
		}
		vLines.push_back(szLine);
	}
	m_fp_in.close();

	string write_file_name = file_path;
	ofstream m_fp_out(const_cast<char *>(write_file_name.c_str()), ios::out | ios::binary);
	if (!m_fp_out.is_open())
	{
		printf("%s open error\n", write_file_name.c_str());
		return CONFIG_OPENFILE_ERROR;
	}

	line_num = vLines.size();
	for (size_t i = 0; i < line_num; i++)
		m_fp_out << vLines[i] << endl;
	m_fp_out.close();

#ifndef WIN32
	system("sync");
#endif

	if (!FileSize(write_file_name.c_str()))
	{
		printf("%s write error!\n", write_file_name.data());
		return CONFIG_OPENFILE_ERROR;
	}

	on_change.clear();

	return CONFIG_SUCCESS;
}

/*************************************************************************
 * Function:			GetKey
 * Descroption:	获取[SECTION]下的某一个键值的字符�?
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 mAttr:	输入参数	主键
 cAttr:		输入参数	子键
 pValue:	输出参数	子键键�?
 * Return:			 CONFIG_RES_E
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
CONFIG_RES_E rmCConfig::GetKey(const char *mAttr, const char *cAttr, char *pValue)
{

	if (m_Map.find(mAttr) == m_Map.end())
	{
		return CONFIG_NO_ATTR;
	}
	else
	{
		KEYMAP mKey = m_Map[mAttr];
		if (mKey.find(cAttr) == mKey.end())
		{
			return CONFIG_NO_ATTR;
		}
		else
		{
			string sTemp = mKey[cAttr].value;
			strcpy(pValue, sTemp.c_str());
		}
	}
	return CONFIG_SUCCESS;
}

/*************************************************************************
 * Function:			GetPos
 * Descroption:	获取[SECTION]下的某一个键值在文件中的位置
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 mAttr:		输入参数	主键
 cAttr:		输入参数	子键
 pos:			输出参数	位置
 * Return:			 CONFIG_RES_E
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
CONFIG_RES_E rmCConfig::GetPos(const char *mAttr, const char *cAttr, int &pos)
{
	if (m_Map.find(mAttr) == m_Map.end())
	{
		return CONFIG_NO_ATTR;
	}
	else
	{
		KEYMAP mKey = m_Map[mAttr];

		if (mKey.find(cAttr) == mKey.end())
		{
			return CONFIG_NO_ATTR;
		}
		else
		{
			pos = mKey[cAttr].line_pos;
		}
	}

	return CONFIG_SUCCESS;
}

/*************************************************************************
 * Function:			InputKey
 * Descroption:	更新已有键�?
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 mAttr:		主键
 cAttr:			子键
 pValue:		更新键�?
 * Return:			 CONFIG_RES_E
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
CONFIG_RES_E rmCConfig::InputKey(const char *mAttr, const char *cAttr, char *pValue)
{
	if (m_Map.find(mAttr) == m_Map.end())
	{
		return CONFIG_NO_ATTR;
	}
	else
	{
		if (m_Map[mAttr].find(cAttr) == m_Map[mAttr].end())
		{
			return CONFIG_NO_ATTR;
		}
		else
		{
			string sTemp(pValue);
			m_Map[mAttr][cAttr].value = sTemp;
			on_change.push_back(m_Map[mAttr][cAttr]);
		}
	}

	return CONFIG_SUCCESS;
}

/*************************************************************************
 * Function:			GetInt
 * Descroption:	获取整形的键�?
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 mAttr:	主键
 cAttr:		子键
 * Return:			 int
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
int rmCConfig::GetInt(const char *mAttr, const char *cAttr)
{
	int nRes = 0;

	memset(m_szKey, 0, sizeof(m_szKey));

	if (CONFIG_SUCCESS == GetKey(mAttr, cAttr, m_szKey))
	{
		nRes = atoi(m_szKey);
		return nRes;
	}
	else
	{
		printf("There is no corresponding KeyValue of [%s][%s]!\n", mAttr, cAttr);
		return CONFIG_ERROR;
	}
}

bool rmCConfig::GetBool(const char *mAttr, const char *cAttr)
{
	bool nRes;

	memset(m_szKey, 0, sizeof(m_szKey));

	if (CONFIG_SUCCESS == GetKey(mAttr, cAttr, m_szKey))
	{	
		if (std::string(m_szKey) == "true")
		{
			return true;
		}
		else if (std::string(m_szKey) == "false")
		{
			return false;
		}
		else
		{
			printf("[%s][%s] should be a bool value \"true\" or \"false\" but got [%s]!\n", mAttr, cAttr, m_szKey);
			return CONFIG_ERROR;
		}
	}
	else
	{
		printf("There is no corresponding KeyValue of [%s][%s]!\n", mAttr, cAttr);
		return CONFIG_ERROR;
	}
}

/*************************************************************************
* Function:			GetFloat
* Descroption:	获取Float的键�?
* Author:				gaoxiang
* History:     <author> <date>  <desc>

* Others:      none
* Parameter:
mAttr:	主键
cAttr:		子键
* Return:			 int
* Support:
ANDROID:
HISI   :
WIN32  :
************************************************************************/
float rmCConfig::GetFloat(const char *mAttr, const char *cAttr)
{
	float nRes = 0;

	memset(m_szKey, 0, sizeof(m_szKey));

	if (CONFIG_SUCCESS == GetKey(mAttr, cAttr, m_szKey))
	{
		nRes = atof(m_szKey);
		return nRes;
	}
	else
	{
		printf("There is no corresponding KeyValue of [%s][%s]!\n", mAttr, cAttr);
		return CONFIG_ERROR;
	}
}

/*************************************************************************
 * Function:			GetStr
 * Descroption:	获取键值的字符�?
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 mAttr:	主键
 cAttr:		子键
 * Return:			 char *
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
std::string rmCConfig::GetStr(const char *mAttr, const char *cAttr)
{
	memset(m_szKey, 0, sizeof(m_szKey));
	string str;

	if (CONFIG_SUCCESS == GetKey(mAttr, cAttr, m_szKey))
	{
		str = m_szKey;
	}
	else
	{
		printf("There is no corresponding KeyValue of [%s][%s]!\n", mAttr, cAttr);
	}

	return str;
}

/*************************************************************************
 * Function:			InputValue
 * Descroption:	更新键值，目前只支持int、float、string类型
 * Author:				gaoxiang
 * History:     <author> <date>  <desc>

 * Others:      none
 * Parameter:
 mAttr:		主键
 cAttr:		子键
 value:		输入键�?
 * Return:			 bool
 * Support:
 ANDROID:
 HISI   :
 WIN32  :
 ************************************************************************/
int rmCConfig::InputValue(const char *mAttr, const char *cAttr, int value)
{
	int m_pos = -1;
	if (CONFIG_SUCCESS != GetPos(mAttr, cAttr, m_pos))
	{
		printf("There is no corresponding KeyValue of [%s][%s]!\n", mAttr, cAttr);
		return -1;
	}

	char str_value[CONFIGLEN] = {0};
	rmSprintf(str_value, "%d", value);
	printf("*******str_value: %s\n", str_value);
	if (InputKey(mAttr, cAttr, str_value))
	{
		printf("Input Key failed!\n");
		return -1;
	}

	return 0;
}

/*************************************************************************
* Function:			InputValue
* Descroption:	更新键值，目前只支持int、float、string类型
* Author:				gaoxiang
* History:     <author> <date>  <desc>

* Others:      none
* Parameter:
mAttr:		主键
cAttr:		子键
value:		输入键�?
* Return:			 bool
* Support:
ANDROID:
HISI   :
WIN32  :
************************************************************************/
int rmCConfig::InputValue(const char *mAttr, const char *cAttr, float value)
{
	int m_pos = -1;
	if (CONFIG_SUCCESS != GetPos(mAttr, cAttr, m_pos))
	{
		printf("There is no corresponding KeyValue of [%s][%s]!\n", mAttr, cAttr);
		return -1;
	}

	char str_value[CONFIGLEN] = {0};
	rmSprintf(str_value, "%f", value);
	if (InputKey(mAttr, cAttr, str_value))
	{
		printf("Input Key failed!\n");
		return -1;
	}

	return 0;
}

/*************************************************************************
* Function:			InputValue
* Descroption:	更新键值，目前只支持int、float、string类型
* Author:				gaoxiang
* History:     <author> <date>  <desc>

* Others:      none
* Parameter:
mAttr:		主键
cAttr:		子键
value:		输入键�?
* Return:			 bool
* Support:
ANDROID:
HISI   :
WIN32  :
************************************************************************/
int rmCConfig::InputValue(const char *mAttr, const char *cAttr, std::string value)
{
	int m_pos = -1;
	if (CONFIG_SUCCESS != GetPos(mAttr, cAttr, m_pos))
	{
		printf("There is no corresponding KeyValue of [%s][%s]!\n", mAttr, cAttr);
		return -1;
	}

	char str_value[CONFIGLEN] = {0};
	rmSprintf(str_value, "%s", value.data());
	if (InputKey(mAttr, cAttr, str_value))
	{
		printf("Input Key failed!\n");
		return -1;
	}

	return 0;
}