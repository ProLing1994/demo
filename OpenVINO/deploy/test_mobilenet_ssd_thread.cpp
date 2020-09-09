#include <iostream>
#include <memory>
#include <mutex>
#include <time.h>
#include <thread>

#include "opencv2/opencv.hpp"

#include "common/utils/csrc/file_system.h"
#include "mobilenet_ssd_detector.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

// Windows
 DEFINE_string(image_folder, "F:/test/images",
   "The folder containing the image data");
 DEFINE_string(model_path, "F:/test/models/ssd_License_plate_mobilenetv2.xml",
   "The network model path");
 DEFINE_string(output_folder, "F:/test/images_result",
   "The folder containing the output results");

// Ubuntu 
//DEFINE_string(image_folder, "/home/huanyuan/code/images",
//  "The folder containing the image data");
//DEFINE_string(model_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.xml",
//  "The network model path");
//DEFINE_string(output_folder, "/home/huanyuan/code/images_result",
//  "The folder containing the output results");

void gen_result(cv::Mat& img_src,
                const std::vector<OPENVINO::ObjectInformation>& objects, 
                const std::string output_image_path) {
  int num_objects = static_cast<int>(objects.size());

  for (int i = 0; i < num_objects; ++i) {
    LOG(INFO) << "location: " << objects[i].location_;
    LOG(INFO) << "label: " << objects[i].name_.c_str() << ", score: " << objects[i].score_ * 100;

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

void ssd_detect_thread(
    const std::vector<std::string>& image_names, 
    const int loop_times, 
    float* time_num) {
	
	std::shared_ptr<OPENVINO::MobilenetSSDDetector> mobilenet_ssd_detector;
	mobilenet_ssd_detector.reset(new OPENVINO::MobilenetSSDDetector());
	int error_int = mobilenet_ssd_detector->init(FLAGS_model_path.c_str());

  for (int i = 0; i < loop_times; i++) {
		for (int idx = 0; idx < image_names.size(); idx++) {
			std::string image_path = FLAGS_image_folder + "/" + image_names[idx];
			std::string output_image_path = FLAGS_output_folder + "/" + image_names[idx];

			cv::Mat img_src = cv::imread(image_path.c_str(), 1);
			std::vector<OPENVINO::ObjectInformation> objects;

			clock_t begin, end;
			begin = clock();
			mobilenet_ssd_detector->detect(img_src, &objects);
			end = clock();
			LOG(INFO) << "time= " << 1.0*(end - begin) / CLOCKS_PER_SEC * 1000.0 << "ms";
			(*time_num) += 1.0*(end - begin) / CLOCKS_PER_SEC * 1000.0;

      gen_result(img_src, objects, output_image_path);
		}	// end for (int idx = 0; idx < image_names.size(); idx++)
  }	// end for (int i = 0; i < loop_times; i++)
}  

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::vector<std::string> image_subfolder;
  std::vector<std::string> image_names;

  yh_common::list_directory(FLAGS_image_folder.c_str(), image_subfolder, image_names);

  int loop_times = 100;

  float time_num_t1 = 0.0;
  float time_num_t2 = 0.0;
  float time_num_t3 = 0.0;
  float time_num_t4 = 0.0;
  std::thread t1(ssd_detect_thread, image_names, loop_times, &time_num_t1);
  std::thread t2(ssd_detect_thread, image_names, loop_times, &time_num_t2);
  std::thread t3(ssd_detect_thread, image_names, loop_times, &time_num_t3);
  std::thread t4(ssd_detect_thread, image_names, loop_times, &time_num_t4);
	
  t1.join();
  t2.join();
  t3.join();
  t4.join();

  LOG(INFO) << "t1 average time= " << time_num_t1 / loop_times / image_names.size() << "ms";
  LOG(INFO) << "t2 average time= " << time_num_t2 / loop_times / image_names.size() << "ms";
  LOG(INFO) << "t3 average time= " << time_num_t3 / loop_times / image_names.size() << "ms";
  LOG(INFO) << "t4 average time= " << time_num_t4 / loop_times / image_names.size() << "ms";

  google::ShutdownGoogleLogging();
  return 0;
}