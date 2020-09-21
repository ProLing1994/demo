#include <fstream>  
#include <memory>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "inference_detection_openvino.hpp"
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
DEFINE_string(yuv_path, "/home/huanyuan/data/yuv/test_license_plate.yuv",
 "The yuv data path");
DEFINE_string(model_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.xml",
 "The network model path");
DEFINE_string(output_folder, "/home/huanyuan/data/images_result",
 "The folder containing the output results");
DEFINE_string(device, "CPU",
 "device name, support for ['CPU'/'GPU']");
DEFINE_int32(nthreads, 4,
 "CPU nthreads");
DEFINE_bool(show_image, true,
 "show image");
DEFINE_bool(output_image, true,
 "output image");

static void DrawRectangle(cv::Mat& cvMatImageSrc,
	const std::vector<inference_openvino::OBJECT_INFO_S>& nObject) {
  int s32ObjectNum = static_cast<int>(nObject.size());

  for (int i = 0; i < s32ObjectNum; ++i) {
	LOG(INFO) << "location: " << nObject[i].cvRectLocation;
	LOG(INFO) << "label: " << nObject[i].strClassName.c_str() << ", score: " << nObject[i].f32Score * 100;

	cv::rectangle(cvMatImageSrc, nObject[i].cvRectLocation, cv::Scalar(0, 0, 255), 2);

	char scText[256];
	sprintf(scText, "%s %.1f%%", nObject[i].strClassName.c_str(), nObject[i].f32Score * 100);

	int s32BaseLine = 0;
	cv::Size label_size = cv::getTextSize(scText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &s32BaseLine);
	cv::putText(cvMatImageSrc, scText, cv::Point(nObject[i].cvRectLocation.x,
	  nObject[i].cvRectLocation.y + label_size.height),
	  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1);
  }
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // model init
	inference_openvino::INFERENCE_OPTIONS_S InferenceOptions;
	InferenceOptions.f64Threshold = 0.5;
  std::vector<std::string> nClassName;
  nClassName.push_back("background");
  nClassName.push_back("License_plate");
	InferenceOptions.nClassName = nClassName;
	InferenceOptions.OpenvinoOptions.u32ThreadNum = FLAGS_nthreads;
	InferenceOptions.OpenvinoOptions.strDeviceType = FLAGS_device;
  std::unique_ptr<inference_openvino::rmInferenceDetectionOpenvino> InferenceDetectionOpenvino;
  InferenceDetectionOpenvino.reset(new inference_openvino::rmInferenceDetectionOpenvino(FLAGS_model_path, InferenceOptions));
  int error_int = InferenceDetectionOpenvino->Init();
	if (error_int < 0) {
		LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Init failed";
		return -1;
	}

  // data init
  const int s32ImageWidth = 2592;
  const int s32ImageHeight = 1920;
  const int s32FrameSize = s32ImageWidth * s32ImageHeight * 3 / 2;
  char* pstscYuvBuf = new char[s32FrameSize];

  std::ifstream Fin;
  Fin.open(FLAGS_yuv_path, std::ios_base::in | std::ios_base::binary);
  if (Fin.fail()) {
	LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", open " << FLAGS_yuv_path << " failed";
	return -1;
  }
  Fin.seekg(0, std::ios::end);
  std::streampos StreamPos = Fin.tellg();
  int s32FrameNum = static_cast<int>(StreamPos / s32FrameSize);
  LOG(INFO) << "[Total] Frame number: " << s32FrameNum;

  Fin.clear();
  Fin.seekg(0, std::ios_base::beg);
  unsigned long long u64StartTime = 0, u64EndTime = 0;
  unsigned long long u64ReadTime = 0, u64CvtColorTime = 0, u64DetectTime = 0;
  unsigned long long u64ReadAverageTime = 0, u64CvtColorAverageTime = 0, u64DetectAverageTime = 0;

  // output init
  std::ofstream Fout;
  if (FLAGS_output_image) {
	std::string strInputPath = FLAGS_yuv_path;
	#ifdef WIN32
	std::string strInputName = strInputPath.substr(strInputPath.find_last_of("\\") + 1);
	std::string strOutputPath = FLAGS_output_folder + "\\result_" + strInputName;
	#else
	std::string strInputName = strInputPath.substr(strInputPath.find_last_of("/") + 1);
	std::string strOutputPath = FLAGS_output_folder + "/result_" + strInputName;
	#endif
	LOG(INFO) << strOutputPath;
	Fout.open(strOutputPath, std::ios_base::out | std::ios_base::binary);
  }

  for (int i = 0; i < s32FrameNum; ++i) {
		TEST_TIME(u64StartTime);

		Fin.read(pstscYuvBuf, s32FrameSize);

		TEST_TIME(u64EndTime);
		u64ReadTime = u64EndTime - u64StartTime;
		u64ReadAverageTime += u64ReadTime;
		TEST_TIME(u64StartTime);

		cv::Mat cvMatYuvImage(s32ImageHeight * 3 / 2, s32ImageWidth, CV_8UC1, static_cast<void*>(pstscYuvBuf));
		cv::Mat cvMatRgbImage;
		cv::cvtColor(cvMatYuvImage, cvMatRgbImage, cv::COLOR_YUV420sp2RGB);
		// cv::cvtColor(yuv_img, rgb_img, cv::COLOR_YUV420sp2BGR);

		TEST_TIME(u64EndTime);
		u64CvtColorTime = u64EndTime - u64StartTime;
		u64CvtColorAverageTime += u64CvtColorTime;
		TEST_TIME(u64StartTime);

		std::vector<inference_openvino::OBJECT_INFO_S> nObject;
		InferenceDetectionOpenvino->Detect(cvMatRgbImage, &nObject);

		TEST_TIME(u64EndTime);
		u64DetectTime = u64EndTime - u64StartTime;
		u64DetectAverageTime += u64DetectTime;

		if (FLAGS_show_image) {
			DrawRectangle(cvMatRgbImage, nObject);
			cv::resize(cvMatRgbImage, cvMatRgbImage, cv::Size(640, 480));
			cv::imshow("image", cvMatRgbImage);
			cv::waitKey(1);
		}

		if (FLAGS_output_image) {
			//gen_result(rgb_img, objects);
			//writer.write(rgb_img);
			//writer << rgb_img;
			DrawRectangle(cvMatYuvImage, nObject);
			Fout.write(reinterpret_cast<char*>(cvMatYuvImage.data), s32FrameSize);
		}
		LOG(INFO) << "\033[0;31mFrame: " << i << ", Read time: " << u64ReadTime << "ms, Cvt color time: " << u64CvtColorTime << "ms, Detect time: " << u64DetectTime << " ms. \033[;39m";
  }

  LOG(INFO) << "\033[0;31mRead average time = " << u64ReadAverageTime / s32FrameNum << "ms, Cvt color average time = " << u64CvtColorAverageTime / s32FrameNum << "ms, Detect average time = " << u64DetectAverageTime / s32FrameNum << "ms. \033[0;39m";
	delete [] pstscYuvBuf;
  Fin.close();
  Fout.close();
  //writer.release();
  google::ShutdownGoogleLogging();
  return 0;
}