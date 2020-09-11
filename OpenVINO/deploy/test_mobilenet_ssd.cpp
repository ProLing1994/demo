#include <iostream>
#include <memory>
#include <time.h>

#include "opencv2/opencv.hpp"

#include "common/utils/csrc/file_system.h"
#include "inference_detection_openvino.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

// Windows
//  DEFINE_string(image_folder, "F:/test/images",
//    "The folder containing the image data");
//  DEFINE_string(model_path, "F:/test/models/ssd_License_plate_mobilenetv2.xml",
//    "The network model path");
//  DEFINE_string(output_folder, "F:/test/images_result",
//    "The folder containing the output results");

// Ubuntu 
DEFINE_string(image_folder, "/home/huanyuan/code/images",
 "The folder containing the image data");
DEFINE_string(model_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.xml",
 "The network model path");
DEFINE_string(output_folder, "/home/huanyuan/code/images_result",
 "The folder containing the output results");

void gen_result(cv::Mat& img_src,
                const std::vector<inference_openvino::OBJECT_INFO_S>& objects, 
                const std::string output_image_path) {
  int num_objects = static_cast<int>(objects.size());

  for (int i = 0; i < num_objects; ++i) {
    LOG(INFO) << "location: " << objects[i].cvRectLocation;
    LOG(INFO) << "label: " << objects[i].strClassName.c_str() << ", score: " << objects[i].f32Score * 100;

    cv::rectangle(img_src, objects[i].cvRectLocation, cv::Scalar(0, 0, 255), 2);

    char text[256];
    sprintf(text, "%s %.1f%%", objects[i].strClassName.c_str(), objects[i].f32Score * 100);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::putText(img_src, text, cv::Point(objects[i].cvRectLocation.x,
        objects[i].cvRectLocation.y + label_size.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1);
  }
  cv::imwrite(output_image_path, img_src);
  // cv::imshow("image", img_src);
  // cv::waitKey(0);
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::shared_ptr<inference_openvino::rmInferenceDetectionOpenvino> mobilenet_ssd_detector;
  mobilenet_ssd_detector.reset(new inference_openvino::rmInferenceDetectionOpenvino(FLAGS_model_path));
  int error_int = mobilenet_ssd_detector->Init();

  std::vector<std::string> image_subfolder;
  std::vector<std::string> image_names;
  yh_common::list_directory(FLAGS_image_folder.c_str(), image_subfolder, image_names);

  double time_num = 0.0;
  int loop_times = 10;

  for (int i = 0; i < loop_times; i++) {
    for (int idx = 0; idx < image_names.size(); idx++) {
      std::string image_path = FLAGS_image_folder + "/" + image_names[idx];
      std::string output_image_path = FLAGS_output_folder + "/" + image_names[idx];

      cv::Mat img_src = cv::imread(image_path.c_str(), 1);
      std::vector<inference_openvino::OBJECT_INFO_S> objects;

      clock_t begin, end;
      begin = clock();
      mobilenet_ssd_detector->Detect(img_src, &objects);
      end = clock();
      LOG(INFO) << "time= " << 1.0*(end - begin) / CLOCKS_PER_SEC * 1000.0 << "ms";
      time_num += 1.0*(end - begin) / CLOCKS_PER_SEC * 1000.0;
      
      gen_result(img_src, objects, output_image_path);
    }	// end for (int idx = 0; idx < image_names.size(); idx++)
  }	// end for (int i = 0; i < loop_times; i++)

  LOG(INFO) << "average time= " << time_num / loop_times / image_names.size() << "ms";

  google::ShutdownGoogleLogging();
  return 0;
}