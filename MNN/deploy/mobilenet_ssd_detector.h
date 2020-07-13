#ifndef _MNN_SRC_MOBILENET_SSD_DETECTOR_H_
#define _MNN_SRC_MOBILENET_SSD_DETECTOR_H_

#include <memory>
#include <vector>
#include <string>

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace MNN {
  struct ObjectInformation {
	std::string name_;
	cv::Rect location_;
	float score_;
  };

  struct OptionsMMN {
	OptionsMMN() :
	  schedule_config_type(MNN_FORWARD_CPU),
	  schedule_config_numThread(4) {}
	MNNForwardType schedule_config_type;
	int schedule_config_numThread;
  };

  struct OptionsMobilenetSSD {
	OptionsMobilenetSSD() {
	  options_mnn_ = OptionsMMN();
	  input_size_ = { 300, 300 };
	  dims_ = { 1, 3, 300, 300 };
	  output_name_ = "detection_out";
	  // class_names = {
	  //     "background", "aeroplane", "bicycle", "bird", "boat",
	  //     "bottle", "bus", "car", "cat", "chair",
	  //     "cow", "diningtable", "dog", "horse",
	  //     "motorbike", "person", "pottedplant",
	  //     "sheep", "sofa", "train", "tvmonitor"
	  // };
	  class_names = {
		  "background", "License_plate" };
	}

	OptionsMMN options_mnn_;
	cv::Size input_size_;
	std::vector<int> dims_;
	// float mean_vals_[3] = { 127.5f, 127.5f, 127.5f };
	// float norm_vals_[3] = { 0.007843f, 0.007843f, 0.007843f };
	float mean_vals_[3] = { 104.0f, 117.0f, 123.0f };
	float norm_vals_[3] = { 1.0f, 1.0f, 1.0f };
	std::string output_name_;
	std::vector<std::string> class_names;
  };

  class MobilenetSSDDetector {
  public:
	MobilenetSSDDetector();
	~MobilenetSSDDetector();
	int init(const char* model_path);
	int detect(const cv::Mat& img_src, std::vector<ObjectInformation>* objects);

  private:
	uint8_t* GetImage(const cv::Mat& img_src) {
	  uchar* data_ptr = new uchar[img_src.total() * 4];
	  cv::Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
	  cv::cvtColor(img_src, img_tmp, CV_BGR2RGBA, 4);
	  return (uint8_t*)img_tmp.data;
	}

  private:
	OptionsMobilenetSSD options_mobilenet_ssd_;
	std::unique_ptr<MNN::Interpreter> mobilenetssd_interpreter_;
	std::shared_ptr<MNN::CV::ImageProcess> image_process = nullptr;
	MNN::Session* mobilenetssd_session_ = nullptr;
	MNN::Tensor * input_tensor_ = nullptr;
  };
}
#endif // _MNN_SRC_MOBILENET_SSD_DETECTOR_H_