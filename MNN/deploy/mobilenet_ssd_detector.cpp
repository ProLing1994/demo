#include <iostream>
#include "mobilenet_ssd_detector.h" 

#include "glog/logging.h"

MNN::MobilenetSSDDetector::MobilenetSSDDetector() {
  options_mobilenet_ssd_ = OptionsMobilenetSSD();
}

MNN::MobilenetSSDDetector::~MobilenetSSDDetector() {
  image_process = nullptr;
  mobilenetssd_session_ = nullptr;
  input_tensor_ = nullptr;
}

int MNN::MobilenetSSDDetector::init(const char* model_path) {
  LOG(INFO) << "start init. " << std::endl;
  mobilenetssd_interpreter_.reset(MNN::Interpreter::createFromFile(model_path));
  if (mobilenetssd_interpreter_ == nullptr) {
	return -1;
  }

  // ScheduleConfig
  MNN::ScheduleConfig schedule_config_;
  schedule_config_.type = options_mobilenet_ssd_.options_mnn_.schedule_config_type;
  schedule_config_.numThread = options_mobilenet_ssd_.options_mnn_.schedule_config_numThread;
  mobilenetssd_session_ = mobilenetssd_interpreter_->createSession(schedule_config_);

  // InputTensor
  input_tensor_ = mobilenetssd_interpreter_->getSessionInput(mobilenetssd_session_, NULL);
  mobilenetssd_interpreter_->resizeTensor(input_tensor_, options_mobilenet_ssd_.dims_);
  mobilenetssd_interpreter_->resizeSession(mobilenetssd_session_);

  // ImageProcess
  MNN::CV::ImageProcess::Config image_config;
  image_config.filterType = MNN::CV::BICUBIC;
  ::memcpy(image_config.mean, options_mobilenet_ssd_.mean_vals_, sizeof(options_mobilenet_ssd_.mean_vals_));
  ::memcpy(image_config.normal, options_mobilenet_ssd_.norm_vals_, sizeof(options_mobilenet_ssd_.norm_vals_));
  image_config.sourceFormat = MNN::CV::RGBA;
  image_config.destFormat = MNN::CV::BGR;
  image_process.reset(MNN::CV::ImageProcess::create(image_config));

  // Matrix
  MNN::CV::Matrix trans;
  trans.setScale(1.0f, 1.0f);
  image_process->setMatrix(trans);

  LOG(INFO) << "end init. ";
  return 0;
}

int MNN::MobilenetSSDDetector::detect(const cv::Mat& img_src, std::vector<ObjectInformation>* objects) {
  LOG(INFO) << "start detect.";

  if (img_src.empty()) {
	LOG(ERROR) << "input empty.";
	return -1;
  }

  int width = img_src.cols;
  int height = img_src.rows;

  // preprocess
  cv::Mat img_resized;
  cv::resize(img_src, img_resized, options_mobilenet_ssd_.input_size_);
  uint8_t* data_ptr = GetImage(img_resized);
  image_process->convert(data_ptr, options_mobilenet_ssd_.input_size_.width, options_mobilenet_ssd_.input_size_.height, 0, input_tensor_);

  // runSession
  mobilenetssd_interpreter_->runSession(mobilenetssd_session_);
  MNN::Tensor* output_tensor = mobilenetssd_interpreter_->getSessionOutput(mobilenetssd_session_, options_mobilenet_ssd_.output_name_.c_str());

  // copy to host
  MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());      // Tensor::CAFFE
  output_tensor->copyToHostTensor(&output_host);

  auto output_ptr = output_host.host<float>();
  for (int i = 0; i < output_host.height(); ++i) {
    int index = i * output_host.width();
    ObjectInformation object;
    object.name_ = options_mobilenet_ssd_.class_names[int(output_ptr[index + 0])];
    object.score_ = output_ptr[index + 1];
    object.location_.x = output_ptr[index + 2] * width;
    object.location_.y = output_ptr[index + 3] * height;
    object.location_.width = output_ptr[index + 4] * width - object.location_.x;
    object.location_.height = output_ptr[index + 5] * height - object.location_.y;
    objects->push_back(object);
  }

  LOG(INFO) << "end detect.";
  return 0;
}

