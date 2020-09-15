#include "inference_detection_model.hpp"

namespace inference_openvino {

	rmInferenceDetectionModel::rmInferenceDetectionModel() {
    LicensePlateModel.reset(nullptr);
    FaceModel.reset(nullptr);
	}

	rmInferenceDetectionModel::~rmInferenceDetectionModel() {
    LicensePlateModel.reset(nullptr);
    FaceModel.reset(nullptr);
	}

  int rmInferenceDetectionModel::InitLicensePlateModel(std::string strModelPath, INFERENCE_OPTIONS_S InferenceOptions) {
    LicensePlateModel.reset(new rmInferenceDetectionOpenvino(strModelPath, InferenceOptions));
    return LicensePlateModel->Init();
  }

  int rmInferenceDetectionModel::InitFaceModel(std::string strModelPath, INFERENCE_OPTIONS_S InferenceOptions) {
    FaceModel.reset(new rmInferenceDetectionOpenvino(strModelPath, InferenceOptions));
    return FaceModel->Init();
  }

  int rmInferenceDetectionModel::DetectLicensePlateModel(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject) {
    if (LicensePlateModel != nullptr) {
      return LicensePlateModel->Detect(cvMatImage, pstnObject);
    }
  }

  int rmInferenceDetectionModel::DetectFaceModel(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject) {
    if (FaceModel != nullptr) {
      return FaceModel->Detect(cvMatImage, pstnObject);
    }
  }

  int rmInferenceDetectionModel::CheckModel() {
    if (FaceModel == nullptr && 
        LicensePlateModel == nullptr) {
      return -1;
    }
    return 0;
  }

}