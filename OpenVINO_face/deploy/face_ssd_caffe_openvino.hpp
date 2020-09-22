#ifndef _RMAI_FACE_SSD_CAFFE_OPENVINO_H
#define _RMAI_FACE_SSD_CAFFE_OPENVINO_H

#include <vector>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "inference_engine.hpp"

struct OBJECT_INFO_S {
OBJECT_INFO_S() :
  cvRectLocation(0, 0, 0, 0),
  strClassName(""),
  f32Score(0.0) {}

cv::Rect cvRectLocation;
std::string strClassName;
float f32Score;
};

struct OPENVINO_SSD_PARAM_S {
    OPENVINO_SSD_PARAM_S() :
	  u32ThreadNum(1),
	  strModelPath(""),
	  f64Threshold(0.5),
	  nClassName() {}

  OPENVINO_SSD_PARAM_S(const OPENVINO_SSD_PARAM_S& params) :
	  u32ThreadNum(params.u32ThreadNum),
	  strModelPath(params.strModelPath),
	  f64Threshold(params.f64Threshold),
	  nClassName(params.nClassName){}

  unsigned int u32ThreadNum; //!< 采用线程数
  std::string strModelPath; //!< .xml模型路径
  double f64Threshold; //!< 检测阈值
	std::vector<std::string> nClassName; //!< 类名
};

class rmInferenceFaceSsdOpenvino {
public:
  rmInferenceFaceSsdOpenvino(const OPENVINO_SSD_PARAM_S& params);
  ~rmInferenceFaceSsdOpenvino();
  int build();
  bool infer(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject);

private:
  OPENVINO_SSD_PARAM_S mParams;
	InferenceEngine::CNNNetwork m_Network;
	std::string m_strInputName;
	InferenceEngine::InputInfo::Ptr m_InputInfo;
  InferenceEngine::SizeVector m_InputDims;
	std::string m_strOutputName;
	InferenceEngine::DataPtr m_OutputInfo;
  InferenceEngine::SizeVector m_OutputDims; 

	InferenceEngine::InferRequest m_InferRrequest;
};


#endif //_RMAI_FACE_SSD_CAFFE_OPENVINO_H