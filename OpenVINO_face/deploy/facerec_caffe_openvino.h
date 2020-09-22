#ifndef _RMAI_FACEREC_CAFFE_OPENVINO_H
#define _RMAI_FACEREC_CAFFE_OPENVINO_H

#include <vector>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "inference_engine.hpp"

struct OpenvinoRecParam
{
	unsigned int u32ThreadNum; //!< 采用线程数

	float pixelMean[3]{127.5f, 127.5f, 127.5f}; //!< 输入均值，BGR顺序
	float pixelScale{0.007843}; //!< 输入方差

	std::string strModelPath; //!< .xml模型路径
	int featureDim{512};
};

class rmInferenceFaceRecOpenvino {
	public:
		rmInferenceFaceRecOpenvino(const OpenvinoRecParam& params);
		~rmInferenceFaceRecOpenvino();
		int build();
		bool infer(const cv::Mat& ipt, std::vector<float>& features);

	private:
		OpenvinoRecParam mParams;
		InferenceEngine::CNNNetwork m_Network;
		std::string m_strInputName;
		InferenceEngine::InputInfo::Ptr m_InputInfo;
		InferenceEngine::SizeVector m_InputDims;
		std::string m_strOutputName;
		InferenceEngine::DataPtr m_OutputInfo;
		InferenceEngine::SizeVector m_OutputDims;


		InferenceEngine::InferRequest m_InferRrequest;
};

#endif //_RMAI_FACEREC_CAFFE_OPENVINO_H