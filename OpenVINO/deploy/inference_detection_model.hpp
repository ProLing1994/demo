#ifndef __AI_INFERENCE_DETECTION_MODEL_H__
#define __AI_INFERENCE_DETECTION_MODEL_H__

#include "inference_detection_openvino.hpp"
#include "RMAI_MOSAIC_API.h"

namespace inference_openvino {

  class rmInferenceDetectionModel{
    public:
      rmInferenceDetectionModel();
      ~rmInferenceDetectionModel();

      int InitLicensePlateModel(std::string strModelPath, INFERENCE_OPTIONS_S InferenceOptions);
      int InitFaceModel(std::string strModelPath, INFERENCE_OPTIONS_S InferenceOptions);
      int DetectLicensePlateModel(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject);
      int DetectFaceModel(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject);
      int CheckModel();
      void UpdateResultNum(const int s32ResultNum);
      void ExpandCapacity();
      int PushResult(const MOSAIC_RESULT_INFO_S& Result, const int s32ResultId);
      void GetResult(MOSAIC_RESULT_INFO_S** nResult);
      void GetResultNum(int* s32ResultNum);
  
    private:
      std::unique_ptr<rmInferenceDetectionOpenvino> m_LicensePlateModel;
      std::unique_ptr<rmInferenceDetectionOpenvino> m_FaceModel;
      MOSAIC_RESULT_INFO_S* m_nResult;
      int m_s32ResultCapacity;
      int m_s32ResultNum;
  };
}

#endif // __AI_INFERENCE_DETECTION_MODEL_H__