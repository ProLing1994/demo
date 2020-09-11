#ifndef __AI_INFERENCE_DETECTION_OPENVINO_H__
#define __AI_INFERENCE_DETECTION_OPENVINO_H__

#include <memory>
#include <vector>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "inference_engine.hpp"

namespace inference_openvino {
	
	struct OBJECT_INFO_S {
		OBJECT_INFO_S() :
			cvRectLocation(0, 0, 0, 0),
			strClassName(""),
			f32Score(0.0) {}

		cv::Rect cvRectLocation;
		std::string strClassName;
		float f32Score;
	};

	struct OPENVINO_OPTIONS_S {
	OPENVINO_OPTIONS_S() :
	  strDeviceName("CPU"),
	  // strDeviceName("GPU"),
	  u32ThreadNum(4) {}
		
	OPENVINO_OPTIONS_S(const OPENVINO_OPTIONS_S& OpenvinoOptions) :
		strDeviceName(OpenvinoOptions.strDeviceName),
	  u32ThreadNum(OpenvinoOptions.u32ThreadNum) {}

	std::string strDeviceName;// ["CPU"/"GPU"/"MULTI:CPU,GPU"]
	unsigned int u32ThreadNum;
  };

	struct INFERENCE_OPTIONS_S {
		INFERENCE_OPTIONS_S():
			OpenvinoOptions(OPENVINO_OPTIONS_S()),
			f64Threshold(0.5) {
			nClassName = {
				"background", "License_plate" };
			// nClassName = {
			//     "background", "aeroplane", "bicycle", "bird", "boat",
			//     "bottle", "bus", "car", "cat", "chair",
			//     "cow", "diningtable", "dog", "horse",
			//     "motorbike", "person", "pottedplant",
			//     "sheep", "sofa", "train", "tvmonitor"
			// };
		}

		INFERENCE_OPTIONS_S(const INFERENCE_OPTIONS_S& InferenceOptions) :
			OpenvinoOptions(InferenceOptions.OpenvinoOptions),
			f64Threshold(InferenceOptions.f64Threshold),
			nClassName(InferenceOptions.nClassName) {}

		OPENVINO_OPTIONS_S OpenvinoOptions;
		double f64Threshold;
		std::vector<std::string> nClassName;
	};

	class rmInferenceDetectionOpenvino {
	public:
		rmInferenceDetectionOpenvino(const std::string strModelPath);
		rmInferenceDetectionOpenvino(const std::string strModelPath, 
																const INFERENCE_OPTIONS_S& InferenceOptions);
		~rmInferenceDetectionOpenvino();
		int Init();
		int Detect(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject);

	private:
		INFERENCE_OPTIONS_S m_InferenceOptions;
		InferenceEngine::CNNNetwork m_Network;
		InferenceEngine::InferRequest m_InferRrequest;
		InferenceEngine::InputInfo::Ptr m_InputInfo;
		InferenceEngine::DataPtr m_OutputInfo;
		std::string m_strInputName;
		std::string m_strOutputName;
		std::string m_strModelPath;
	};
}
#endif // __AI_INFERENCE_DETECTION_OPENVINO_H__