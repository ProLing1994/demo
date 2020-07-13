#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"

#include "common/utils/csrc/file_system.h"
#include "mobilenet_ssd_detector.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(image_folder, "/home/huanyuan/code/MobileNet-SSD/images",
  "The folder containing the image data");
DEFINE_string(model_path, "/home/huanyuan/code/models/mobilenet_iter_73000.caffe.mnn", 
  "The network model path");
DEFINE_string(output_folder, "/home/huanyuan/code/MobileNet-SSD/images_result", 
  "The folder containing the output results");

int main(int argc, char* argv[]){
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  
  std::shared_ptr<MNN::MobilenetSSDDetector> mobilenet_ssd_detector;
  mobilenet_ssd_detector.reset(new MNN::MobilenetSSDDetector());
  int error_int = mobilenet_ssd_detector->init(FLAGS_model_path.c_str());

  std::vector<std::string> image_subfolder;
  std::vector<std::string> image_names;

  yh_common::list_directory(FLAGS_image_folder.c_str(), image_subfolder, image_names);

  for(int idx=0; idx<image_names.size(); idx++){
    std::string image_path = FLAGS_image_folder + "/" + image_names[idx];
    std::string output_image_path = FLAGS_output_folder + "/" + image_names[idx];

    cv::Mat img_src = cv::imread(image_path.c_str(), 1);
    std::vector<MNN::ObjectInformation> objects;
    mobilenet_ssd_detector->detect(img_src, &objects);

    int num_objects = static_cast<int>(objects.size());
    for (int i = 0; i < num_objects; ++i) {
      std::cout << "location: " << objects[i].location_ << std::endl;
      std::cout << "label: " << objects[i].name_.c_str() << ", score: "<< objects[i].score_ * 100 << std::endl;

      cv::rectangle(img_src, objects[i].location_, cv::Scalar(0, 0, 255), 2);

      char text[256];
      sprintf(text, "%s %.1f%%", objects[i].name_.c_str(), objects[i].score_ * 100);

      int baseLine = 0;
      cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

      cv::putText(img_src, text, cv::Point(objects[i].location_.x,
        objects[i].location_.y + label_size.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1);
    }
    cv::imwrite(output_image_path, img_src);
    // cv::imshow("image", img_src);
    // cv::waitKey(0);
  }

  google::ShutdownGoogleLogging();
  return 0;
}