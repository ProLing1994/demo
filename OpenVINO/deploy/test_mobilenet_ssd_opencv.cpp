#include <iostream>
#include <memory>
#include <time.h>

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

#include "common/utils/csrc/file_system.h"
#include "mobilenet_ssd_detector.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#ifndef HAVE_INF_ENGINE
#define HAVE_INF_ENGINE
#endif

// Ubuntu 
DEFINE_string(image_folder, "/home/huanyuan/code/images",
  "The folder containing the image data");
DEFINE_string(model_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.caffemodel",
  "The network model path");
DEFINE_string(prototxt_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2_fpn_ncnn_concat.prototxt",
  "The network prototxt path");
DEFINE_string(output_folder, "/home/huanyuan/code/images_result",
  "The folder containing the output results");

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

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // std::string binFileName = FLAGS_model_path.substr(0, FLAGS_model_path.rfind('.')) + ".bin";
  // cv::dnn::Net net = cv::dnn::readNetFromModelOptimizer(FLAGS_model_path, binFileName);

  cv::dnn::Net net = cv::dnn::readNetFromCaffe(FLAGS_prototxt_path, FLAGS_model_path);
  if (net.empty())
  {
      std::cerr << "Can't load network by using the following files: " << std::endl;
      return -1;
  }
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
  // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  OPENVINO::OptionsMobilenetSSD ssd;

  std::vector<std::string> image_subfolder;
  std::vector<std::string> image_names;
  yh_common::list_directory(FLAGS_image_folder.c_str(), image_subfolder, image_names);

  float time_num = 0.0;
  int loop_times = 100;

  for (int i = 0; i < loop_times; i++) {
    for (int idx = 0; idx < image_names.size(); idx++) {
      std::string image_path = FLAGS_image_folder + "/" + image_names[idx];
      std::string output_image_path = FLAGS_output_folder + "/" + image_names[idx];
      
      cv::Mat img_src = cv::imread(image_path.c_str(), 1);
      std::vector<OPENVINO::ObjectInformation> objects;

      int width = img_src.cols;
		  int height = img_src.rows;

      clock_t begin, end;
      begin = clock();

      cv::Mat inputBlob = cv::dnn::blobFromImage(img_src, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);
      // cv::Mat inputBlob = cv::dnn::blobFromImage(img_src, 1.0, cv::Size(300, 300));
      net.setInput(inputBlob, "data");
      cv::Mat detection = net.forward("detection_out");

      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

      for (int i = 0; i < detectionMat.rows; i++) { 

        float confidence = detectionMat.at<float>(i, 2);
        int label = static_cast<int>(detectionMat.at<float>(i, 1));
        int xmin = static_cast<int>(detectionMat.at<float>(i, 3) * width);
        int ymin = static_cast<int>(detectionMat.at<float>(i, 4) * height);
        int xmax = static_cast<int>(detectionMat.at<float>(i, 5) * width);
        int ymax = static_cast<int>(detectionMat.at<float>(i, 6) * height);


        if (confidence > ssd.threshold) {
          /** Drawing only objects with >50% probability **/
          OPENVINO::ObjectInformation object;
          object.name_ = ssd.class_names[label];
          object.score_ = confidence;
          object.location_.x = xmin;
          object.location_.y = ymin;
          object.location_.width = xmax - xmin;
          object.location_.height = ymax - ymin;
          objects.push_back(object);
        }
      }

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