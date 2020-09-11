#ifndef __RMAI_MOSACI_API_H__
#define __RMAI_MOSACI_API_H__

#include <vector>

typedef struct
{
    int s32Type;//0:车牌   1:人脸   2:车牌+人脸
    char scLicensePlate[128];//车牌模型路径
    char scFace[128];//人脸模型路径
    char scReserve[512];
}CONFIG_FILE_S;

typedef struct
{
    int s32GPU;//0:CPU  1:GPU
    int s32ThreadNum;//线程数量
    char scReserve[512];
}CONFIG_INFO_S;

typedef struct
{
    unsigned long long u64PTS;
    int s32Format;//0:yuv420sp/nv12   1:GRB
    int s32Width;
    int s32Height;
    unsigned long long u64Viraddr[3];
    unsigned long long u64PhyAddr[3];
    char scReserve[64];
}IMAG_INFO_S;

typedef struct
{
    char scReserve[128];
}INPUT_INFO_S;

typedef struct
{
    int s32Type;//0:车牌 1:人脸
    int as32Rect[4];//0:x 1:y 2:width 3:height
    char scReserve[32];
}RESULT_INFO_S;

/*
*初始化
*/
int Init(CONFIG_FILE_S* pstFile,CONFIG_INFO_S* pstInfo, int* handle);

/*
*检测
*/
int Run(int handle,IMAG_INFO_S* pstImage,INPUT_INFO_S* pstInput, std::vector<RESULT_INFO_S>& nResult);

/*
*去初始化
*/
int UnInit(int handle);

#endif // __RMAI_MOSACI_API_H__