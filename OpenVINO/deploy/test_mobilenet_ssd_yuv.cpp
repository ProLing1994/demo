#include <fstream>  
#include <memory>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "mobilenet_ssd_detector.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#ifdef WIN32
#ifndef TEST_TIME
#include <time.h>
#define TEST_TIME(times) do{\
        clock_t cur_time;\
	    cur_time = clock();\
	    times = static_cast<unsigned long long>(1000.0 * cur_time / CLOCKS_PER_SEC) ;\
	}while(0)
#endif
#else
#ifndef TEST_TIME
#include <sys/time.h>
#define TEST_TIME(times) do{\
        struct timeval cur_time;\
	    gettimeofday(&cur_time, NULL);\
	    times = (cur_time.tv_sec * 1000000llu + cur_time.tv_usec) / 1000llu;\
	}while(0)
#endif
#endif

// Windows
// DEFINE_string(yuv_path, "F:\\test\\DataSets\\test_001.yuv",
//   "The yuv data path");
// DEFINE_string(model_path, "F:\\test\\models\\ssd_License_plate_mobilenetv2.xml",
//   "The network model path");
// DEFINE_string(output_folder, "F:\\test\\images_result",
//   "The folder containing the output results");
// DEFINE_string(device, "CPU",
//   "device name, support for ['CPU'/'GPU']");
// DEFINE_int32(nthreads, 4,
//   "CPU nthreads");
// DEFINE_bool(show_image, true,
//   "show image");
// DEFINE_bool(output_image, true,
//   "output image");

// Ubuntu 
DEFINE_string(yuv_path, "/home/huanyuan/code/yuv/output.yuv",
 "The yuv data path");
DEFINE_string(model_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.xml",
 "The network model path");
DEFINE_string(output_folder, "/home/huanyuan/code/images_result",
 "The folder containing the output results");
DEFINE_string(device, "CPU",
 "device name, support for ['CPU'/'GPU']");
DEFINE_int32(nthreads, 4,
 "CPU nthreads");
DEFINE_bool(show_image, true,
 "show image");
DEFINE_bool(output_image, true,
 "output image");

void gen_result(cv::Mat& img_src,
	const std::vector<OPENVINO::ObjectInformation>& objects) {
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
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // model init
  double threshold = 0.5;
  std::vector<std::string> class_names;
  class_names.push_back("background");
  class_names.push_back("License_plate");
  std::shared_ptr<OPENVINO::MobilenetSSDDetector> mobilenet_ssd_detector;
  // mobilenet_ssd_detector.reset(new OPENVINO::MobilenetSSDDetector());
  mobilenet_ssd_detector.reset(new OPENVINO::MobilenetSSDDetector(threshold, class_names, FLAGS_device, FLAGS_nthreads));
  int error_int = mobilenet_ssd_detector->init(FLAGS_model_path);

  // data init
  int image_width = 2592;
  int image_height = 1920;
  const int framesize = image_width * image_height * 3 / 2;
  char* yuv_buf = new char[framesize];

  std::ifstream fin;
  fin.open(FLAGS_yuv_path, std::ios_base::in | std::ios_base::binary);
  if (fin.fail()) {
	LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", open " << FLAGS_yuv_path << " failed";
	return -1;
  }
  fin.seekg(0, std::ios::end);
  std::streampos ps = fin.tellg();
  int FrameCount = static_cast<int>(ps / framesize);
  LOG(INFO) << "[Total] Frame number: " << FrameCount;

  fin.clear();
  fin.seekg(0, std::ios_base::beg);
  unsigned long long start_time = 0, end_time = 0;
  unsigned long long read_time = 0, cvt_color_time = 0, detect_time = 0;
  unsigned long long read_average_time = 0, cvt_color_average_time = 0, detect_average_time = 0;

  // output init
  //cv::VideoWriter writer;
  std::ofstream fout;
  if (FLAGS_output_image) {
	std::string input_path = FLAGS_yuv_path;
	#ifdef WIN32
	std::string input_name = input_path.substr(input_path.find_last_of("\\") + 1);
	//input_name.replace(input_name.find(".yuv"), 4, ".avi");
	std::string output_path = FLAGS_output_folder + "\\result_" + input_name;
	#else
	std::string input_name = input_path.substr(input_path.find_last_of("/") + 1);
	//input_name.replace(input_name.find(".yuv"), 4, ".avi");
	std::string output_path = FLAGS_output_folder + "/result_" + input_name;
	#endif
	LOG(INFO) << output_path;
	fout.open(output_path, std::ios_base::out | std::ios_base::binary);
	//writer.open(output_path, writer.fourcc('F', 'L', 'V', '1'), 25, cv::Size(image_width, image_height), true);
  }

  for (int i = 0; i < FrameCount; ++i)
  {
	TEST_TIME(start_time);

	fin.read(yuv_buf, framesize);

	TEST_TIME(end_time);
	read_time = end_time - start_time;
	read_average_time += read_time;
	TEST_TIME(start_time);

	cv::Mat yuv_img(image_height * 3 / 2, image_width, CV_8UC1, static_cast<void*>(yuv_buf));
	cv::Mat rgb_img;
	cv::cvtColor(yuv_img, rgb_img, cv::COLOR_YUV420sp2RGB);
	// cv::cvtColor(yuv_img, rgb_img, cv::COLOR_YUV420sp2BGR);

	TEST_TIME(end_time);
	cvt_color_time = end_time - start_time;
	cvt_color_average_time += cvt_color_time;
	TEST_TIME(start_time);

	std::vector<OPENVINO::ObjectInformation> objects;
	mobilenet_ssd_detector->detect(rgb_img, &objects);

	TEST_TIME(end_time);
	detect_time = end_time - start_time;
	detect_average_time += detect_time;

	if (FLAGS_show_image) {
	  gen_result(rgb_img, objects);
	  cv::resize(rgb_img, rgb_img, cv::Size(640, 480));
	  cv::imshow("image", rgb_img);
	  cv::waitKey(1);
	}

	if (FLAGS_output_image) {
	  //gen_result(rgb_img, objects);
	  //writer.write(rgb_img);
	  //writer << rgb_img;
	  gen_result(yuv_img, objects);
	  fout.write(reinterpret_cast<char*>(yuv_img.data), framesize);
	}
	LOG(INFO) << "\033[0;31mFrame: " << i << ", Read time: " << read_time << "ms, Cvt color time: " << cvt_color_time << "ms, Detect time: " << detect_time << " ms. \033[;39m";
  }

  LOG(INFO) << "\033[0;31mRead average time = " << read_average_time / FrameCount << "ms, Cvt color average time = " << cvt_color_average_time / FrameCount << "ms, Detect average time = " << detect_average_time / FrameCount << "ms. \033[0;39m";
  fin.close();
  fout.close();
  //writer.release();
  google::ShutdownGoogleLogging();
  return 0;
}