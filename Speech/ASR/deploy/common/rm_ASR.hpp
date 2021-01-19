#ifndef _RM_ASR_API_H__
#define _RM_ASR_API_H__

int RMAPI_AI_AsrInit(char* file_path);
int RMAPI_AI_AsrAlgStart(short* pBuffer, int length, char* outKeyword);
int RMAPI_AI_AsrDeinit();

#endif // _RM_ASR_API_H__