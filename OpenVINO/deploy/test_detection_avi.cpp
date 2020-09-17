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

// // Windows
// DEFINE_string(avi_path, "F:\\test\\yuv\\test_taxi_face.avi",
//   "The yuv data path");
// DEFINE_int32(avi_imwidth, 1280,
//   "The yuv data width");
// DEFINE_int32(avi_imheight, 720,
//   "The yuv data height");
// DEFINE_string(model_path, "F:\\test\\models\\ssd_face_mask.xml",
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
DEFINE_string(avi_path, "/home/huanyuan/code/yuv/test_taxi_face.avi",
"The yuv data path");
DEFINE_int32(avi_imwidth, 1280,
"The yuv data width");
DEFINE_int32(avi_imheight, 720,
"The yuv data height");
DEFINE_string(model_path, "/home/huanyuan/code/models/ssd_face_mask.xml",
"The network model path");
DEFINE_string(output_folder, "/home/huanyuan/code/yuv",
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
    const std::vector<inference_openvino::OBJECT_INFO_S>& nObject,
    const int s32OriImagewWidth,
    const int s32OriImageHeight) {
  int s32ObjectNum = static_cast<int>(nObject.size());
  int s32SrcWidth = 0;
  int s32SrcHeight = 0;
  if (cvMatImageSrc.type() != CV_8UC3) {
    s32SrcWidth = cvMatImageSrc.cols;
    s32SrcHeight = cvMatImageSrc.rows * 2 / 3;
  }
  else {
    s32SrcWidth = cvMatImageSrc.cols;
    s32SrcHeight = cvMatImageSrc.rows;
  }

  for (int i = 0; i < s32ObjectNum; ++i) {
    LOG(INFO) << "location: " << nObject[i].cvRectLocation;
    LOG(INFO) << "label: " << nObject[i].strClassName.c_str() << ", score: " << nObject[i].f32Score * 100;
    
    cv::Rect cvRectLocation;  
    cvRectLocation.x = static_cast<double>(nObject[i].cvRectLocation.x) * s32SrcWidth / s32OriImagewWidth;
    cvRectLocation.y = static_cast<double>(nObject[i].cvRectLocation.y) * s32SrcHeight / s32OriImageHeight;
    cvRectLocation.width = static_cast<double>(nObject[i].cvRectLocation.width) * s32SrcWidth / s32OriImagewWidth;
    cvRectLocation.height = static_cast<double>(nObject[i].cvRectLocation.height) * s32SrcHeight / s32OriImageHeight;

    cv::rectangle(cvMatImageSrc, cvRectLocation, cv::Scalar(0, 0, 255), 2);

    char scText[256];
    sprintf(scText, "%s %.1f%%", nObject[i].strClassName.c_str(), nObject[i].f32Score * 100);

    int s32BaseLine = 0;
    cv::Size label_size = cv::getTextSize(scText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &s32BaseLine);
    cv::putText(cvMatImageSrc, scText, cv::Point(cvRectLocation.x,
      cvRectLocation.y + label_size.height),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1);
    }
}

static void CreateMosaicImage(cv::Mat& cvMatImageSrc,
	  const std::vector<inference_openvino::OBJECT_INFO_S>& nObject,
    const int s32OriImagewWidth,
    const int s32OriImageHeight,
    int s32MosaicSize = 10) {
  int s32ObjectNum = static_cast<int>(nObject.size());
  int s32SrcWidth = 0;
  int s32SrcHeight = 0;
  if (cvMatImageSrc.type() != CV_8UC3) {
    s32SrcWidth = cvMatImageSrc.cols;
    s32SrcHeight = cvMatImageSrc.rows * 2 / 3;
  }
  else {
    s32SrcWidth = cvMatImageSrc.cols;
    s32SrcHeight = cvMatImageSrc.rows;
  }

  for (int i = 0; i < s32ObjectNum; ++i) {
    cv::Rect cvRectLocation;
    cvRectLocation.x = static_cast<double>(nObject[i].cvRectLocation.x) * s32SrcWidth / s32OriImagewWidth;
    cvRectLocation.y = static_cast<double>(nObject[i].cvRectLocation.y) * s32SrcHeight / s32OriImageHeight;
    cvRectLocation.width = static_cast<double>(nObject[i].cvRectLocation.width) * s32SrcWidth / s32OriImagewWidth;
    cvRectLocation.height = static_cast<double>(nObject[i].cvRectLocation.height) * s32SrcHeight / s32OriImageHeight;
    
    for (int s32PixelIdx = cvRectLocation.x; s32PixelIdx < cvRectLocation.x + cvRectLocation.width ; s32PixelIdx += s32MosaicSize) {
      for (int s32PixelIdy = cvRectLocation.y; s32PixelIdy < cvRectLocation.y + cvRectLocation.height; s32PixelIdy += s32MosaicSize) {
        int s32MosaicWidth = 
            s32PixelIdx + s32MosaicSize < cvRectLocation.x + cvRectLocation.width ? s32MosaicSize : cvRectLocation.x + cvRectLocation.width - s32PixelIdx;
        int s32MosaicHeight = 
            s32PixelIdy + s32MosaicSize < cvRectLocation.y + cvRectLocation.height ? s32MosaicSize : cvRectLocation.y + cvRectLocation.height - s32PixelIdy;
        cv::Rect cvRectMosaicRoi = cv::Rect(s32PixelIdx, s32PixelIdy, s32MosaicWidth, s32MosaicHeight);
        cv::Mat cvMatMosaicRoi = cvMatImageSrc(cvRectMosaicRoi);
        if (cvMatImageSrc.type() != CV_8UC3) {
          cv::Scalar scScalarColor = cv::Scalar(cvMatImageSrc.at<uchar>(s32PixelIdy, s32PixelIdx));
          cvMatMosaicRoi.setTo(scScalarColor);
        }
        else {
          cv::Scalar scScalarColor = cv::Scalar(cvMatImageSrc.at<cv::Vec3b>(s32PixelIdy, s32PixelIdx)[0], \
                                      cvMatImageSrc.at<cv::Vec3b>(s32PixelIdy, s32PixelIdx)[1], \
                                      cvMatImageSrc.at<cv::Vec3b>(s32PixelIdy, s32PixelIdx)[2]);
          cvMatMosaicRoi.setTo(scScalarColor);
        }
      }
    }
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
  cv::VideoCapture cap;
  cap.open(FLAGS_avi_path);

  // output init
  cv::VideoWriter writer;
  std::string strOutputPath;
  if (FLAGS_output_image) {
    std::string strInputPath = FLAGS_avi_path;

    #ifdef WIN32
      std::string strInputName = strInputPath.substr(strInputPath.find_last_of("\\") + 1);
      strOutputPath = FLAGS_output_folder + "\\result_" + strInputName;
      strOutputPath.replace(strOutputPath.find(".avi"), 4, "_0.avi");
    #else
      std::string strInputName = strInputPath.substr(strInputPath.find_last_of("/") + 1);
      strOutputPath = FLAGS_output_folder + "/result_" + strInputName;
    #endif
    
	writer.open(strOutputPath, writer.fourcc('M', 'J', 'P', 'G'), 25, cv::Size(FLAGS_avi_imwidth, FLAGS_avi_imheight), true);
	LOG(INFO) << "Output file is stored to " << strOutputPath;
  }

  // Run
  cv::Mat cvMatFrame;
	int s32FrameNum = 0;
  unsigned long long u64StartTime = 0, u64EndTime = 0;
  unsigned long long u64ReadTime = 0, u64CvtColorTime = 0, u64DetectTime = 0;
  unsigned long long u64ReadAverageTime = 0, u64CvtColorAverageTime = 0, u64DetectAverageTime = 0;

  while (1) {

	TEST_TIME(u64StartTime);

	cap >> cvMatFrame;
	s32FrameNum += 1;

	// Stop the program if reached end of video
	if (cvMatFrame.empty())
	{
	  LOG(INFO) << "Done processing !!!";
	  LOG(INFO) << "Output file is stored to " << strOutputPath;
	  // cv::waitKey(3000);
	  break;
	}

	TEST_TIME(u64EndTime);
	u64ReadTime = u64EndTime - u64StartTime;
	u64ReadAverageTime += u64ReadTime;
	TEST_TIME(u64StartTime);

	std::vector<inference_openvino::OBJECT_INFO_S> nObject;
	InferenceDetectionOpenvino->Detect(cvMatFrame, &nObject);

	TEST_TIME(u64EndTime);
	u64DetectTime = u64EndTime - u64StartTime;
	u64DetectAverageTime += u64DetectTime;

	if (FLAGS_show_image) {
	  cv::Mat cvMatResizeImage;
	  cv::resize(cvMatFrame, cvMatResizeImage, cv::Size(640, 480));
	  CreateMosaicImage(cvMatResizeImage, nObject, FLAGS_avi_imwidth, FLAGS_avi_imheight, 15);
	  DrawRectangle(cvMatResizeImage, nObject, FLAGS_avi_imwidth, FLAGS_avi_imheight);
	  cv::imshow("image", cvMatResizeImage);
	  cv::waitKey(1);
	}

	if (FLAGS_output_image) {
	  CreateMosaicImage(cvMatFrame, nObject, FLAGS_avi_imwidth, FLAGS_avi_imheight, 15);
	  DrawRectangle(cvMatFrame, nObject, FLAGS_avi_imwidth, FLAGS_avi_imheight);
	  writer.write(cvMatFrame);
	  // writer << cvMatFrame;
	}
	LOG(INFO) << "\033[0;31mFrame: " << s32FrameNum << ", Read time: " << u64ReadTime << "ms, Cvt color time: " << u64CvtColorTime << "ms, Detect time: " << u64DetectTime << " ms. \033[;39m";
  }

  LOG(INFO) << "\033[0;31mRead average time = " << u64ReadAverageTime / s32FrameNum << "ms, Cvt color average time = " << u64CvtColorAverageTime / s32FrameNum << "ms, Detect average time = " << u64DetectAverageTime / s32FrameNum << "ms. \033[0;39m";
  writer.release();
  google::ShutdownGoogleLogging();
  return 0;
}