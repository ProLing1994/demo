#include <fstream>  
#include <memory>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "RMAI_MOSAIC_API.hpp"
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

//// Ubuntu 
//DEFINE_string(yuv_path, "/home/huanyuan/code/yuv/output.yuv",
// "The yuv data path");
//DEFINE_int32(yuv_imwidth, 2592,
// "The yuv data width");
//DEFINE_int32(yuv_imheight, 1920,
// "The yuv data height");
//DEFINE_int32(model_type, 2,
// "model type, 0: License_plate, 1: face, 2: License_plate&face");
//DEFINE_string(license_plate_model_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.xml",
// "The network model path");
//DEFINE_string(face_model_path, "/home/huanyuan/code/models/ssd_face_mask.xml",
// "The network model path");
//DEFINE_string(output_folder, "/home/huanyuan/code/images_result",
// "The folder containing the output results");
//DEFINE_string(device, "CPU",
// "device name, support for ['CPU'/'GPU']");
//DEFINE_int32(nthreads, 4,
// "CPU nthreads");
//DEFINE_bool(show_image, true,
// "show image");
//DEFINE_bool(output_image, true,
// "output image");

// Win 
DEFINE_string(yuv_path, "F:\\test\\DataSets\\test_001.yuv",
  "The yuv data path");
DEFINE_int32(yuv_imwidth, 2592,
  "The yuv data width");
DEFINE_int32(yuv_imheight, 1920,
  "The yuv data height");
DEFINE_int32(model_type, 2,
  "model type, 0: License_plate, 1: face, 2: License_plate&face");
DEFINE_string(license_plate_model_path, "F:\\test\\models\\ssd_License_plate_mobilenetv2.xml",
  "The network model path");
DEFINE_string(face_model_path, "F:\\test\\models\\ssd_face_mask.xml",
  "The network model path");
DEFINE_string(output_folder, "F:\\test\\images_result",
  "The folder containing the output results");
DEFINE_string(device, "CPU",
  "device name, support for ['CPU'/'GPU']");
DEFINE_int32(nthreads, 4,
  "CPU nthreads");
DEFINE_bool(show_image, true,
  "show image");
DEFINE_bool(output_image, true,
  "output image");

 void DrawRectangle(cv::Mat& cvMatImageSrc,
	const std::vector<RESULT_INFO_S>& nResult) {
  int s32ObjectNum = static_cast<int>(nResult.size());

  for (int i = 0; i < s32ObjectNum; ++i) {
    cv::Rect cvRectLocation;
    cvRectLocation.x = nResult[i].as32Rect[0];
    cvRectLocation.y = nResult[i].as32Rect[1];
    cvRectLocation.width = nResult[i].as32Rect[2];
    cvRectLocation.height = nResult[i].as32Rect[3];
    
    std::string label = nResult[i].s32Type == 0 ?  "License_plate" : "License_plate";
    LOG(INFO) << "location: " << cvRectLocation;
    LOG(INFO) << "label: " << label;

    cv::rectangle(cvMatImageSrc, cvRectLocation, cv::Scalar(0, 0, 255), 2);

    char scText[256];
    sprintf(scText, "%s", label.c_str());

    int s32BaseLine = 0;
    cv::Size label_size = cv::getTextSize(scText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &s32BaseLine);
    cv::putText(cvMatImageSrc, scText, cv::Point(cvRectLocation.x,
      cvRectLocation.y + label_size.height),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1);
  }
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Init
  CONFIG_FILE_S ConfigFile;
  ConfigFile.s32ModelType = FLAGS_model_type;
  strcpy(ConfigFile.scLicensePlate, FLAGS_license_plate_model_path.c_str());
  strcpy(ConfigFile.scFace, FLAGS_face_model_path.c_str());
  strcpy(ConfigFile.scReserve, "");

  CONFIG_INFO_S ConfigInfo;
  if (FLAGS_device == "CPU") {
    ConfigInfo.s32GPU = 0;
    ConfigInfo.s32ThreadNum = FLAGS_nthreads;
    }
  else if (FLAGS_device == "GPU") {
    ConfigInfo.s32GPU = 1;
    ConfigInfo.s32ThreadNum = 0;
  }
  else {
    ConfigInfo.s32GPU = -1;
    ConfigInfo.s32ThreadNum = 0;
  }
  void* pstModel = nullptr;
  int error_int = Init(&ConfigFile, &ConfigInfo, &pstModel);
	if (error_int != 0) {
		LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Init failed";
		return -1;
	}

  // data init
  const int s32ImageWidth = FLAGS_yuv_imwidth;
  const int s32ImageHeight = FLAGS_yuv_imheight;
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
  unsigned long long u64ReadTime = 0, u64DetectTime = 0;
  unsigned long long u64ReadAverageTime = 0, u64DetectAverageTime = 0;

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
	LOG(INFO) << "Output path: "<< strOutputPath;
	Fout.open(strOutputPath, std::ios_base::out | std::ios_base::binary);
  }

  // Run
  for (int i = 0; i < s32FrameNum; ++i) {
    TEST_TIME(u64StartTime);

    Fin.read(pstscYuvBuf, s32FrameSize);

    TEST_TIME(u64EndTime);
    u64ReadTime = u64EndTime - u64StartTime;
    u64ReadAverageTime += u64ReadTime;
    TEST_TIME(u64StartTime);

    IMAG_INFO_S ImageInfo;
    ImageInfo.u64PTS = 0;// TODO
    ImageInfo.s32Format = 0;
    ImageInfo.s32Width = s32ImageWidth;
    ImageInfo.s32Height = s32ImageHeight;
    strcpy(ImageInfo.scReserve, "");
    ImageInfo.scViraddr = new char[s32FrameSize];
    strcpy(ImageInfo.scViraddr, pstscYuvBuf);

    INPUT_INFO_S InputInfo;
    strcpy(InputInfo.scReserve, "");

    std::vector<RESULT_INFO_S> nResult;
    error_int = Run(pstModel, &ImageInfo, &InputInfo, nResult);
    if (error_int != 0) {
      LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Run failed";
      return -1;
    }

    TEST_TIME(u64EndTime);
    u64DetectTime = u64EndTime - u64StartTime;
    u64DetectAverageTime += u64DetectTime;

    if (FLAGS_show_image) {
      cv::Mat cvMatYuvImage(s32ImageHeight * 3 / 2, s32ImageWidth, CV_8UC1, static_cast<void*>(pstscYuvBuf));
      cv::Mat cvMatRgbImage;
      cv::cvtColor(cvMatYuvImage, cvMatRgbImage, cv::COLOR_YUV420sp2RGB);
      DrawRectangle(cvMatRgbImage, nResult);
      cv::resize(cvMatRgbImage, cvMatRgbImage, cv::Size(640, 480));
      cv::imshow("image", cvMatRgbImage);
      cv::waitKey(1);
    }

    if (FLAGS_output_image) {
      //gen_result(rgb_img, objects);
      //writer.write(rgb_img);
      //writer << rgb_img;
      cv::Mat cvMatYuvImage(s32ImageHeight * 3 / 2, s32ImageWidth, CV_8UC1, static_cast<void*>(pstscYuvBuf));
      DrawRectangle(cvMatYuvImage, nResult);
      Fout.write(reinterpret_cast<char*>(cvMatYuvImage.data), s32FrameSize);
    }
    LOG(INFO) << "\033[0;31mFrame: " << i << ", Read time: " << u64ReadTime << "ms, Detect time: " << u64DetectTime << " ms. \033[;39m";
    delete [] ImageInfo.scViraddr;
  }

  LOG(INFO) << "\033[0;31mRead average time = " << u64ReadAverageTime / s32FrameNum << "ms, Detect average time = " << u64DetectAverageTime / s32FrameNum << "ms. \033[0;39m";
  delete [] pstscYuvBuf;
  error_int = UnInit(pstModel);
  if (error_int != 0) {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Run failed";
    return -1;
  }
  Fin.close();
  Fout.close();
  // writer.release();
  google::ShutdownGoogleLogging();
  return 0;
}