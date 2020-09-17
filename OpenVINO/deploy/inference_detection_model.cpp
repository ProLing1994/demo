#include "inference_detection_model.hpp"

namespace inference_openvino {

	rmInferenceDetectionModel::rmInferenceDetectionModel() {
    m_LicensePlateModel.reset(nullptr);
    m_FaceModel.reset(nullptr);
    m_s32ResultNum = 0;
    m_s32ResultCapacity = 16;
    m_nResult = new MOSAIC_RESULT_INFO_S[m_s32ResultCapacity];
	}

	rmInferenceDetectionModel::~rmInferenceDetectionModel() {
    m_LicensePlateModel.reset(nullptr);
    m_FaceModel.reset(nullptr);
    delete [] m_nResult;
	}

  int rmInferenceDetectionModel::InitLicensePlateModel(std::string strModelPath, INFERENCE_OPTIONS_S InferenceOptions) {
    m_LicensePlateModel.reset(new rmInferenceDetectionOpenvino(strModelPath, InferenceOptions));
    return m_LicensePlateModel->Init();
  }

  int rmInferenceDetectionModel::InitFaceModel(std::string strModelPath, INFERENCE_OPTIONS_S InferenceOptions) {
    m_FaceModel.reset(new rmInferenceDetectionOpenvino(strModelPath, InferenceOptions));
    return m_FaceModel->Init();
  }

  int rmInferenceDetectionModel::DetectLicensePlateModel(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject) {
    if (m_LicensePlateModel != nullptr) {
      return m_LicensePlateModel->Detect(cvMatImage, pstnObject);
    }
  }

  int rmInferenceDetectionModel::DetectFaceModel(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject) {
    if (m_FaceModel != nullptr) {
      return m_FaceModel->Detect(cvMatImage, pstnObject);
    }
  }

  int rmInferenceDetectionModel::CheckModel() {
    if (m_FaceModel == nullptr && 
        m_LicensePlateModel == nullptr) {
      return -1;
    }
    return 0;
  }
  
  void rmInferenceDetectionModel::UpdateResultNum(const int s32ResultNum) {
    m_s32ResultNum = s32ResultNum;
  }

  void rmInferenceDetectionModel::ExpandCapacity() {
    if (m_s32ResultNum <= m_s32ResultCapacity)
      return;

    delete [] m_nResult;
    m_s32ResultCapacity *= 2;
    m_nResult = new MOSAIC_RESULT_INFO_S[m_s32ResultCapacity];
    return;
  }

  int rmInferenceDetectionModel::PushResult(const MOSAIC_RESULT_INFO_S& Result, const int s32ResultId) {
    if (s32ResultId >= m_s32ResultNum || s32ResultId >= m_s32ResultCapacity)
      return -1;

    m_nResult[s32ResultId] = Result;
    return 0;
  }

  void rmInferenceDetectionModel::GetResult(MOSAIC_RESULT_INFO_S** nResult) {
    *nResult = m_nResult;
  };

  void rmInferenceDetectionModel::GetResultNum(int* s32ResultNum){
    *s32ResultNum = m_s32ResultNum;
  };

}