#include <fstream>  
#include <memory>
#include <mutex>
#include <iostream>
#include <thread>

#include "opencv2/opencv.hpp"

#include "RMAI_MOSAIC_API.h"
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
//DEFINE_string(data_path_0, "/home/huanyuan/data/yuv/test_license_plate.yuv",
//"The yuv data path");
//DEFINE_string(data_path_1, "/home/huanyuan/data/yuv/test_taxi_face.yuv",
//"The yuv data path");
//DEFINE_int32(imwidth_0, 2592,
//"The yuv data width");
//DEFINE_int32(imheight_0, 1920,
//"The yuv data height");
//DEFINE_int32(imwidth_1, 1280,
//"The yuv data width");
//DEFINE_int32(imheight_1, 720,
//"The yuv data height");
//DEFINE_int32(model_type, 2,
//"model type, 0: License_plate, 1: face, 2: License_plate&face");
//DEFINE_string(license_plate_model_path, "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.xml",
//"The network model path");
//DEFINE_string(face_model_path, "/home/huanyuan/code/models/ssd_face_mask.xml",
//"The network model path");
//DEFINE_string(output_folder, "/home/huanyuan/data/images_result",
//"The folder containing the output results");
//DEFINE_string(device, "CPU",
//"device name, support for ['CPU'/'GPU']");
//DEFINE_int32(nthreads, 4,
//"CPU nthreads");
//DEFINE_bool(show_image, true,
//"show image");

// Win
DEFINE_string(data_path_0, "F:\\test\\yuv\\test_license_plate.yuv",
"The yuv data path");
DEFINE_string(data_path_1, "F:\\test\\yuv\\test_taxi_face.yuv",
"The yuv data path");
DEFINE_int32(imwidth_0, 2592,
"The yuv data width");
DEFINE_int32(imheight_0, 1920,
"The yuv data height");
DEFINE_int32(imwidth_1, 1280,
"The yuv data width");
DEFINE_int32(imheight_1, 720,
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

static std::mutex mtx;

static void DrawRectangle(cv::Mat& cvMatImageSrc,
    const MOSAIC_RESULT_INFO_S* nResult,
    const int s32ResultNum,
    const int s32OriImagewWidth,
    const int s32OriImageHeight) {
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

  for (int i = 0; i < s32ResultNum; ++i) {
    cv::Rect cvRectLocation;
    cvRectLocation.x = static_cast<int>(static_cast<double>(nResult[i].as32Rect[0]) * s32SrcWidth / s32OriImagewWidth);
    cvRectLocation.y = static_cast<int>(static_cast<double>(nResult[i].as32Rect[1]) * s32SrcHeight / s32OriImageHeight);
    cvRectLocation.width = static_cast<int>(static_cast<double>(nResult[i].as32Rect[2]) * s32SrcWidth / s32OriImagewWidth);
    cvRectLocation.height = static_cast<int>(static_cast<double>(nResult[i].as32Rect[3]) * s32SrcHeight / s32OriImageHeight);
    
    std::string label = nResult[i].s32Type == 0 ?  "License_plate" : "Face";
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

static void CreateMosaicImage(cv::Mat& cvMatImageSrc,
	  const MOSAIC_RESULT_INFO_S* nResult,
    const int s32ResultNum,
    const int s32OriImagewWidth,
    const int s32OriImageHeight,
    int s32MosaicSize = 10) {
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

  for (int Objectidx = 0; Objectidx < s32ResultNum; ++Objectidx) {
    cv::Rect cvRectLocation;
    cvRectLocation.x = static_cast<int>(static_cast<double>(nResult[Objectidx].as32Rect[0]) * s32SrcWidth / s32OriImagewWidth);
    cvRectLocation.y = static_cast<int>(static_cast<double>(nResult[Objectidx].as32Rect[1]) * s32SrcHeight / s32OriImageHeight);
    cvRectLocation.width = static_cast<int>(static_cast<double>(nResult[Objectidx].as32Rect[2]) * s32SrcWidth / s32OriImagewWidth);
    cvRectLocation.height = static_cast<int>(static_cast<double>(nResult[Objectidx].as32Rect[3]) * s32SrcHeight / s32OriImageHeight);
    
    for (int i = cvRectLocation.x; i < cvRectLocation.x + cvRectLocation.width ; i += s32MosaicSize) {
      for (int j = cvRectLocation.y; j < cvRectLocation.y + cvRectLocation.height; j += s32MosaicSize) {
        int s32MosaicWidth = 
            i + s32MosaicSize < cvRectLocation.x + cvRectLocation.width ? s32MosaicSize : cvRectLocation.x + cvRectLocation.width - i;
        int s32MosaicHeight = 
            j + s32MosaicSize < cvRectLocation.y + cvRectLocation.height ? s32MosaicSize : cvRectLocation.y + cvRectLocation.height - j;
        cv::Rect cvRectMosaicRoi = cv::Rect(i, j, s32MosaicWidth, s32MosaicHeight);
        cv::Mat cvMatMosaicRoi = cvMatImageSrc(cvRectMosaicRoi);
        if (cvMatImageSrc.type() != CV_8UC3) {
          cv::Scalar scScalarColor = cv::Scalar(cvMatImageSrc.at<uchar>(j, i));
          cvMatMosaicRoi.setTo(scScalarColor);
        }
        else {
          cv::Scalar scScalarColor = cv::Scalar(cvMatImageSrc.at<cv::Vec3b>(j, i)[0], \
                                      cvMatImageSrc.at<cv::Vec3b>(j, i)[1], \
                                      cvMatImageSrc.at<cv::Vec3b>(j, i)[2]);
          cvMatMosaicRoi.setTo(scScalarColor);
        }
      }
    }
  }
}

static void ThreaadApi(void* pstModel, std::string image_path, int imwidth, int imheight, std::string image_name) {
  // data init
  const int s32ImageWidth = imwidth;
  const int s32ImageHeight = imheight;
  std::string strInputPath = image_path;
  int s32DataType = 0;

  cv::VideoCapture cap;

  std::ifstream Fin;
  int s32FrameSize;
  char* pstscYuvBuf;

  if (strInputPath.find(".avi") != strInputPath.npos||
      strInputPath.find(".mp4") != strInputPath.npos) {
    cap.open(strInputPath);
    s32DataType = 1;
  }
  else if (strInputPath.find(".yuv") != strInputPath.npos) {
    s32FrameSize = s32ImageWidth * s32ImageHeight * 3 / 2;
    pstscYuvBuf = new char[s32FrameSize];
    
    Fin.open(strInputPath, std::ios_base::in | std::ios_base::binary);
    if (Fin.fail()) {
      LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", open " << strInputPath << " failed";
      return;
    }
    s32DataType = 0;
  }
  else {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Unknown  data type" << image_path;
  }

  // Run
  unsigned long long u64StartTime = 0, u64EndTime = 0;
  unsigned long long u64ReadTime = 0, u64DetectTime = 0;
  unsigned long long u64ReadAverageTime = 0, u64DetectAverageTime = 0;

  cv::Mat cvMatFrame;
  int s32FrameId = 0;

  while (1)  {
    TEST_TIME(u64StartTime);

    if (s32DataType) {
      cap >> cvMatFrame;
      // Stop the program if reached end of video
      if (cvMatFrame.empty()) {
        LOG(INFO) << "Done processing !!!";
        break;
      }
    }
    else {
      Fin.read(pstscYuvBuf, s32FrameSize);
      // Stop the program if reached end of video
      if (Fin.eof()) {
        LOG(INFO) << "Done processing !!!";
        break;
      }
    }

    TEST_TIME(u64EndTime);
    u64ReadTime = u64EndTime - u64StartTime;
    u64ReadAverageTime += u64ReadTime;
    TEST_TIME(u64StartTime);

    MOSAIC_IMAGE_INFO_S ImageInfo;
    ImageInfo.u64PTS = 0;// TODO
    ImageInfo.s32Format = s32DataType;
    ImageInfo.s32Width = s32ImageWidth;
    ImageInfo.s32Height = s32ImageHeight;
    strcpy(ImageInfo.scReserve, "");
    
    if (s32DataType) {
      ImageInfo.scViraddr = new char[s32ImageWidth * s32ImageHeight * 3];
    memcpy(ImageInfo.scViraddr, static_cast<char*>(static_cast<void*>(cvMatFrame.data)), s32ImageHeight * s32ImageWidth * 3);
    }
    else {
      ImageInfo.scViraddr = new char[s32FrameSize];
      strcpy(ImageInfo.scViraddr, pstscYuvBuf);
    }
    
    MOSAIC_INPUT_INFO_S InputInfo;
    strcpy(InputInfo.scReserve, "");

    MOSAIC_RESULT_INFO_S* nResult = nullptr;
    int s32ResultNum = 0;
    int error_int = RMAPI_AI_MOSAIC_RUN(pstModel, &ImageInfo, &InputInfo, &nResult, &s32ResultNum);
    if (error_int != 0) {
      LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Run failed";
      return;
    }

    TEST_TIME(u64EndTime);
    u64DetectTime = u64EndTime - u64StartTime;
    u64DetectAverageTime += u64DetectTime;
    
    if (FLAGS_show_image) {
      cv::Mat cvMatRgbImage;
      if (s32DataType) {
        cv::resize(cvMatFrame, cvMatRgbImage, cv::Size(640, 480));
      }
      else {
        cv::Mat cvMatYuvImage(s32ImageHeight * 3 / 2, s32ImageWidth, CV_8UC1, static_cast<void*>(pstscYuvBuf));
        cv::cvtColor(cvMatYuvImage, cvMatRgbImage, cv::COLOR_YUV420sp2RGB);
        cv::resize(cvMatRgbImage, cvMatRgbImage, cv::Size(640, 480));
      }
      CreateMosaicImage(cvMatRgbImage, nResult, s32ResultNum, s32ImageWidth, s32ImageHeight, 15);
      DrawRectangle(cvMatRgbImage, nResult, s32ResultNum, s32ImageWidth, s32ImageHeight);
      
      mtx.lock();
      cv::imshow(image_name, cvMatRgbImage);
      cv::waitKey(1);
      mtx.unlock();
    }

    LOG(INFO) << "\033[0;31mFrame: " << s32FrameId << ", Read time: " << u64ReadTime << "ms, Detect time: " << u64DetectTime << " ms. \033[;39m";
    s32FrameId += 1;
    delete [] ImageInfo.scViraddr;
  }
  if (s32FrameId) {
    LOG(INFO) << "\033[0;31m[Total] Frame number = " << s32FrameId << ", Read average time = " << u64ReadAverageTime / s32FrameId
      << "ms, Detect average time = " << u64DetectAverageTime / s32FrameId << "ms. \033[0;39m";
    }
  else {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Can not load data";
    return;
  }
  if (!s32DataType) {
    delete [] pstscYuvBuf;
  }
  Fin.close();
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Init
  MOSAIC_CONFIG_FILE_S ConfigFile;
  ConfigFile.s32ModelType = FLAGS_model_type;
  strcpy(ConfigFile.scLicensePlate, FLAGS_license_plate_model_path.c_str());
  strcpy(ConfigFile.scFace, FLAGS_face_model_path.c_str());
  strcpy(ConfigFile.scReserve, "");

  MOSAIC_CONFIG_INFO_S ConfigInfo;
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
  int error_int = RMAPI_AI_MOSAIC_INIT(&ConfigFile, &ConfigInfo, &pstModel);
	if (error_int != 0) {
		LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Init failed";
		return -1;
	}

  // threaad_api(pstModel);
  std::thread t1(ThreaadApi, pstModel, FLAGS_data_path_0, FLAGS_imwidth_0, FLAGS_imheight_0, "license_plate");
  std::thread t2(ThreaadApi, pstModel, FLAGS_data_path_1, FLAGS_imwidth_1, FLAGS_imheight_1, "face");

  t1.join();
  t2.join();

  error_int = RMAPI_AI_MOSAIC_UNINIT(&pstModel);
  if (error_int != 0) {
    LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Run failed";
    return -1;
  }
  google::ShutdownGoogleLogging();
  return 0;
}