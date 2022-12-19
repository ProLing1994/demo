
#pragma once
#ifndef _READ_CONFIG_H_
#define _READ_CONFIG_H_

#include <map>
#include <fstream>
#include <string>
#include <vector>

#define CONFIGLEN 512

/*
用法：比如读�?
[Section1]
key1 = 1
key2 = abcdw
[Section2]
key1 = 3
key2 = ddba

rmCConfig  ini;
ini.OpenFile("./Test.ini", "r");
char *pVal1 = ini.GetStr("Section1", "key2");
int  nKey = ini.GetInt("Section2", "key1");
*/

enum CONFIG_RES_E
{
	CONFIG_SUCCESS = 0,			//成功
	CONFIG_ERROR = -1,			//普通错�?
	CONFIG_OPENFILE_ERROR = -2, //打开文件失败
	CONFIG_SYNC_ERROR = -3,
	CONFIG_NO_ATTR = 1 //无对应的键�?
};

//键值与位置
typedef struct KEY_VALUE_S
{
	std::string value;
	unsigned int line_pos;
} KeyValue;

//              子键索引    子键�?
typedef std::map<std::string, KeyValue> KEYMAP;
//              主键索引 主键�?
typedef std::map<std::string, KEYMAP> MAINKEYMAP;
//				键值数�?
typedef std::vector<KeyValue> KEYVECTOR;
// config 文件的基本操作类

class rmCConfig
{
public:
	// 构造函�?
	rmCConfig();
	rmCConfig(const char *pathName);

	// 析够函数
	virtual ~rmCConfig();

public:
	//获取整形的键�?
	int GetInt(const char *mAttr, const char *cAttr);
	//获取Float的键�?
	float GetFloat(const char *mAttr, const char *cAttr);
	//获取键值的字符�?
	std::string GetStr(const char *mAttr, const char *cAttr);

	// 输入参数
	int InputValue(const char *mAttr, const char *cAttr, int value);
	int InputValue(const char *mAttr, const char *cAttr, float value);
	int InputValue(const char *mAttr, const char *cAttr, std::string value);

	// 打开config 文件
	CONFIG_RES_E OpenFile(const char *pathName, bool isSync = false);
	// 写入config文件
	CONFIG_RES_E WriteFile();

protected:
	// 读取config文件
	CONFIG_RES_E GetKey(const char *mAttr, const char *cAttr, char *value);

	//获取key在文件中位置
	CONFIG_RES_E GetPos(const char *mAttr, const char *cAttr, int &pos);

	//修改已有键�?
	CONFIG_RES_E InputKey(const char *mAttr, const char *cAttr, char *value);

	void KeySort(KEYVECTOR &v, int left, int right);

	int FileSize(const char *filename);

	bool CopyFile(const char *srcFile, const char *dstFile);

private:
	std::string file_path;
	char m_szKey[CONFIGLEN];
	MAINKEYMAP m_Map;
	KEYVECTOR on_change;
};

#endif // _READ_CONFIG_H_