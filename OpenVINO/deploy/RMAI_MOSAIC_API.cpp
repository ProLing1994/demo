#include <string>

#include "inference_detection_model.hpp"
#include "RMAI_MOSAIC_API.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

/*
*初始化
*/
int Init(CONFIG_FILE_S* pstFile, CONFIG_INFO_S* pstInfo, void** Handle){
  CHECK_NOTNULL(pstFile);
  CHECK_NOTNULL(pstInfo);

  // Check input
  if (pstFile->s32ModelType != 0 &&
      pstFile->s32ModelType != 1 &&
      pstFile->s32ModelType != 2) {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ 
      << ", ModelType only support for ['0:license_plate', '1:face', '2:license_plate+face'], can not be " 
      << pstFile->s32ModelType;
    return -1;
  }

  if (pstInfo->s32GPU != 0 &&
      pstInfo->s32GPU != 1) {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ 
      << ", DeviceName only support for ['0:CPU', '1:GPU']";
    return -1;
  }

  // init
  *Handle = new inference_openvino::rmInferenceDetectionModel();
  inference_openvino::rmInferenceDetectionModel* pstInferenceModels = static_cast<inference_openvino::rmInferenceDetectionModel*>(*Handle);
  // static std::unique_ptr<inference_openvino::rmInferenceDetectionModel> pstInferenceModels;
  // pstInferenceModels.reset(new inference_openvino::rmInferenceDetectionModel());
  // *Handle = static_cast<void*>(&pstInferenceModels);

  inference_openvino::INFERENCE_OPTIONS_S InferenceOptions;
  std::vector<std::string> nClassName;
  std::string strModelPath;
  int s32ErrorCode;

  if(pstFile->s32ModelType == 0 || pstFile->s32ModelType == 2) {
    InferenceOptions.f64Threshold = 0.5;
    nClassName.clear();
    nClassName.push_back("background");
    nClassName.push_back("License_plate");
    InferenceOptions.nClassName = nClassName;
	  InferenceOptions.OpenvinoOptions.strDeviceType = pstInfo->s32GPU == 0 ? "CPU" : "GPU";
    InferenceOptions.OpenvinoOptions.u32ThreadNum = static_cast<unsigned int>(pstInfo->s32ThreadNum);
    strModelPath = pstFile->scLicensePlate;
    s32ErrorCode = pstInferenceModels -> InitLicensePlateModel(strModelPath, InferenceOptions);
    if (s32ErrorCode < 0) {
      LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Init License Plate Model failed";
      return -1;
	  }
  }

  if(pstFile->s32ModelType == 1 || pstFile->s32ModelType == 2) {
    InferenceOptions.f64Threshold = 0.5;
    nClassName.clear();
    nClassName.push_back("background");
    nClassName.push_back("Face");
    InferenceOptions.nClassName = nClassName;
	  InferenceOptions.OpenvinoOptions.strDeviceType = pstInfo->s32GPU == 0 ? "CPU" : "GPU";
    InferenceOptions.OpenvinoOptions.u32ThreadNum = static_cast<unsigned int>(pstInfo->s32ThreadNum);
    strModelPath = pstFile->scFace;
    s32ErrorCode = pstInferenceModels -> InitFaceModel(strModelPath, InferenceOptions);
    if (s32ErrorCode < 0) {
      LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Init Face Model failed";
      return -1;
	  }
  }
  return 0;
}

/*
*检测
*/
int Run(void* Handle, IMAG_INFO_S* pstImage, INPUT_INFO_S* pstInput, std::vector<RESULT_INFO_S>& nResult){
  CHECK_NOTNULL(Handle);
  CHECK_NOTNULL(pstImage);
  CHECK_NOTNULL(pstInput);

  // Check input
  if (pstImage->s32Format != 0 &&
      pstImage->s32Format != 1) {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ 
      << ", Imgae format only support for ['0:yuv420sp', '1:RGB'], can not be " 
      << pstImage->s32Format;
    return -1;
  }

  int s32ImageHeight = pstImage->s32Height;
  int s32ImageWidth = pstImage->s32Width;
  if (s32ImageHeight <= 0 ||
      s32ImageWidth <= 0) {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ 
      << ", Imgae height and width must > 0";
    return -1;
  }
  
  inference_openvino::rmInferenceDetectionModel *pstInferenceModels = static_cast<inference_openvino::rmInferenceDetectionModel* >(Handle);
  int s32ErrorCode = pstInferenceModels->CheckModel();
  // std::unique_ptr<inference_openvino::rmInferenceDetectionModel> *pstInferenceModels = static_cast<std::unique_ptr<inference_openvino::rmInferenceDetectionModel>* >(Handle);
  // int s32ErrorCode = (*pstInferenceModels)->CheckModel();
  if (s32ErrorCode !=0) {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ 
      << ", Do not find License Plate Model or Face Model, please init License Plate Model or Face Model first";
  }

  // run
  // data preparation
  cv::Mat cvMatRgbImage(s32ImageHeight, s32ImageWidth, CV_8UC3);
  if (pstImage->s32Format == 0) {
    cv::Mat cvMatYuvImage(s32ImageHeight * 3 / 2, s32ImageWidth, CV_8UC1, static_cast<void*>(pstImage->scViraddr));
    cv::cvtColor(cvMatYuvImage, cvMatRgbImage, cv::COLOR_YUV420sp2RGB);
  }
  else {
    cvMatRgbImage.data = static_cast<uchar*>(static_cast<void*>(pstImage->scViraddr));
  }

  // model detect
  std::vector<inference_openvino::OBJECT_INFO_S> nObject;
  pstInferenceModels->DetectLicensePlateModel(cvMatRgbImage, &nObject);
  pstInferenceModels->DetectFaceModel(cvMatRgbImage, &nObject);
  // (*pstInferenceModels)->DetectLicensePlateModel(cvMatRgbImage, &nObject);
  // (*pstInferenceModels)->DetectFaceModel(cvMatRgbImage, &nObject);
  
  // gen nResult
  for (int s32IdObject = 0; s32IdObject < nObject.size(); s32IdObject++) {
    RESULT_INFO_S Result_Info;

    if (nObject[s32IdObject].strClassName == "License_plate") Result_Info.s32Type = 0;
    else if (nObject[s32IdObject].strClassName == "Face") Result_Info.s32Type = 1;
    else {
      LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ 
        << ", Object Type Error, only support for ['License_plate', 'Face'], can not be " << nObject[s32IdObject].strClassName;
      return -1;
    } 

    Result_Info.as32Rect[0] = nObject[s32IdObject].cvRectLocation.x;
    Result_Info.as32Rect[1] = nObject[s32IdObject].cvRectLocation.y;
    Result_Info.as32Rect[2] = nObject[s32IdObject].cvRectLocation.width;
    Result_Info.as32Rect[3] = nObject[s32IdObject].cvRectLocation.height;

    nResult.push_back(Result_Info);
  }
  return 0;
}

/*
*去初始化
*/
int UnInit(void* Handle) {
  CHECK_NOTNULL(Handle);
  inference_openvino::rmInferenceDetectionModel *pstInferenceModels = static_cast<inference_openvino::rmInferenceDetectionModel* >(Handle);
  pstInferenceModels->~rmInferenceDetectionModel();
  // std::unique_ptr<inference_openvino::rmInferenceDetectionModel> *pstInferenceModels = static_cast<std::unique_ptr<inference_openvino::rmInferenceDetectionModel>* >(Handle);
  // (*pstInferenceModels)->~rmInferenceDetectionModel();
  return 0;
}
