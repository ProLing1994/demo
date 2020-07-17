#ifndef _OPENVIVO_DEPLOY_MOBILENET_SSD_DETECTOR_H_
#define _OPENVIVO_DEPLOY_MOBILENET_SSD_DETECTOR_H_

#include <memory>
#include <vector>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "inference_engine.hpp"

namespace OPENVINO {
	
	struct ObjectInformation {
		std::string name_;
		cv::Rect location_;
		float score_;
	};

	struct OptionsOPENVINO {
	OptionsOPENVINO() :
	  device_name("CPU"),
		nthreads(1) {}

	// device_name, support for ["CPU"/"GPU"/"MULTI:CPU,GPU"]
	std::string device_name;
	int nthreads;
  };

	struct OptionsMobilenetSSD {
		OptionsMobilenetSSD() {
			// class_names = {
			//     "background", "aeroplane", "bicycle", "bird", "boat",
			//     "bottle", "bus", "car", "cat", "chair",
			//     "cow", "diningtable", "dog", "horse",
			//     "motorbike", "person", "pottedplant",
			//     "sheep", "sofa", "train", "tvmonitor"
			// };
			threshold = 0.5;
			class_names = {
				"background", "License_plate" };
		}
		OptionsOPENVINO options_openvino_;
		double threshold;
		std::vector<std::string> class_names;
	};

	class MobilenetSSDDetector {
	public:
		MobilenetSSDDetector();
		~MobilenetSSDDetector();
		int init(const std::string model_path);
		int detect(const cv::Mat& img_src, std::vector<ObjectInformation>* objects);

	private:
		OptionsMobilenetSSD options_mobilenet_ssd_;
		InferenceEngine::CNNNetwork network_;
		InferenceEngine::InferRequest infer_request_;
		InferenceEngine::InputInfo::Ptr input_info_;
		InferenceEngine::DataPtr output_info_;
		std::string input_name_;
		std::string output_name_;
	};
}
#endif // _OPENVIVO_DEPLOY_MOBILENET_SSD_DETECTOR_H_