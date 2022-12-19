#ifndef __RMAI_MOSACI_API_H__
#define __RMAI_MOSACI_API_H__

#ifdef __cplusplus 
extern "C" {
#endif

  /*! \brief 模型配置结构体
  *  s32ModelType:              加载模型类型：0：仅加载车牌模型，输出车牌结果
  *                                          1：仅加载人脸模型，输出人脸结果
  *                                          2：加载车牌和人脸模型，输出车牌和人脸结果
  *  scLicensePlate[128]:       车牌模型路径
  *  scFace[128]:               人脸模型路径s
  *  scReserve[512]:            预留字段
  */
  typedef struct
  {
      int s32ModelType;//0:车牌  1:人脸  2:车牌+人脸
      char scLicensePlate[128];//车牌模型路径
      char scFace[128];//人脸模型路径
      char scReserve[512];
  }MOSAIC_CONFIG_FILE_S;

  /*! \brief 设备配置结构体
  *  s32GPU:                    模型推理运算资源类型：0：使用 CPU 进行模型推理
  *                                                 1：使用 GPU 进行模型推理，仅支持集成显卡
  *  s32ThreadNum:              线程数，在使用 CPU 时的多线程推理，使用 GPU 无效
  *  scReserve[512]:            预留字段
  */
  typedef struct
  {
      int s32GPU;//0:CPU  1:GPU
      int s32ThreadNum;//线程数量
      char scReserve[512];
  }MOSAIC_CONFIG_INFO_S;

  /*! \brief 输入图像信息结构体
  *  u64PTS:                    帧时间戳
  *  s32Format:                 图像数据类型：0：yuv420sp
  *                                          1：RGB 
  *  s32Width:                  图像宽度
  *  s32Height:                 图像高度
  *  scViraddr:                 图像数据内容 
  *  scReserve[64]:             预留字段
  */
  typedef struct
  {
      unsigned long long u64PTS;
      int s32Format;//0:yuv420sp/nv12  1:RGB
      int s32Width;
      int s32Height;
      char* scViraddr;//数据内容
      char scReserve[64];
  }MOSAIC_IMAGE_INFO_S;

  /*! \brief 输入预留字段结构体
  *  scReserve[128]:            预留字段
  */
  typedef struct
  {
      char scReserve[128];
  }MOSAIC_INPUT_INFO_S;

  /*! \brief 输出检测结果信息结构体
  *  s32Type:                    检测结果类型：0：车牌
  *                                           1：人脸
  *  as32Rect[4]:                检测框位置：0：图像左上角横坐标 x
  *                                         1：图像左上角纵坐标 y
  *                                         2. 图像宽
  *                                         3. 图像高
  *  scReserve[32]:              预留字段
  */
  typedef struct
  {
      int s32Type;//0:车牌  1:人脸
      int as32Rect[4];//0:x  1:y  2:width  3:height
      char scReserve[32];
  }MOSAIC_RESULT_INFO_S;

  /*! \brief 算法初始化
  *  \param[in]  pstFile:        模型配置结构体
  *  \param[in]  pstInfo:        设备配置结构体
  *  \param[out] Handle:         初始化模型指针的指针
  *  \return     0 表示成功，其他表示失败
  */
  int RMAPI_AI_MOSAIC_INIT(MOSAIC_CONFIG_FILE_S* pstFile,
                           MOSAIC_CONFIG_INFO_S* pstInfo, 
                           void** Handle);

  /*! \brief 运行检测算法，返回结果
  *  \param[in]  Handle:        模型指针
  *  \param[in]  pstImage:      输入图像信息结构体
  *  \param[in]  pstInput:      输入预留字段结构体
  *  \param[out] nResult:       输出检测结果信息结构体
  *  \param[out] s32ResultNum:  输出检测结果结构体的个数
  *  \return     0 表示成功，其他表示失败
  */
  int RMAPI_AI_MOSAIC_RUN(void* Handle,
                          MOSAIC_IMAGE_INFO_S* pstImage, 
                          MOSAIC_INPUT_INFO_S* pstInput, 
                          MOSAIC_RESULT_INFO_S** nResult, 
                          int* s32ResultNum);

  /*! \brief 算法析构
  *  \param[in]  Handle:        模型指针
  *  \return     0 表示成功，其他表示失败
  */
  int RMAPI_AI_MOSAIC_UNINIT(void** Handle);

#ifdef __cplusplus 
}
#endif

#endif // __RMAI_MOSACI_API_H__ 