#ifndef __AI_INFERENCE_DETECTION_MODEL_H__
#define __AI_INFERENCE_DETECTION_MODEL_H__

#include "inference_detection_openvino.hpp"

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
  
    private:
      std::unique_ptr<rmInferenceDetectionOpenvino> LicensePlateModel;
      std::unique_ptr<rmInferenceDetectionOpenvino> FaceModel;
  };
}

#endif // __AI_INFERENCE_DETECTION_MODEL_H__