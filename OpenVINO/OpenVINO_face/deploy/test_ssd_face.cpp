#include <fstream>  
#include <memory>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "face_ssd_caffe_openvino.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

// Ubuntu 
DEFINE_string(face_path, "/home/huanyuan/data/images/face_id.jpg",
 "The yuv data path");
DEFINE_string(model_path, "/home/huanyuan/code/models/face/ssd_face_model/RFB-VGG-FaceV3-20191230-45.xml",
 "The network model path");
DEFINE_string(output_folder, "/home/huanyuan/data/images_result",
 "The folder containing the output results");
DEFINE_int32(nthreads, 4,
 "CPU nthreads");
DEFINE_bool(show_image, true,  
 "show image");
DEFINE_bool(output_image, true,
 "output image");

static void DrawRectangle(cv::Mat& cvMatImageSrc,
	const std::vector<OBJECT_INFO_S>& nObject) {
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
	OPENVINO_SSD_PARAM_S InferenceOptions;
	InferenceOptions.f64Threshold = 0.5;
  std::vector<std::string> nClassName;
  nClassName.push_back("background");
  nClassName.push_back("face");
	InferenceOptions.nClassName = nClassName;
	InferenceOptions.u32ThreadNum = FLAGS_nthreads;
	InferenceOptions.strModelPath = FLAGS_model_path;
  std::unique_ptr<rmInferenceFaceSsdOpenvino> InferenceDetectionOpenvino;
  InferenceDetectionOpenvino.reset(new rmInferenceFaceSsdOpenvino(InferenceOptions));
  int error_int = InferenceDetectionOpenvino->build();
	if (error_int < 0) {
		LOG(ERROR) << "ERROR, func: " << __FUNCTION__ << ", line: " << __LINE__ << ", Init failed";
		return -1;
	}

	cv::Mat cvMatRgbImage;
	std::string strInputPath = FLAGS_face_path;
	cvMatRgbImage = cv::imread(strInputPath);

	std::vector<OBJECT_INFO_S> nObject;
	InferenceDetectionOpenvino->infer(cvMatRgbImage, &nObject);

	if (FLAGS_show_image) {
		DrawRectangle(cvMatRgbImage, nObject);
		// cv::resize(cvMatRgbImage, cvMatRgbImage, cv::Size(640, 480));
		cv::imshow("image", cvMatRgbImage);
		cv::waitKey();
	}

	if (FLAGS_output_image) {
		std::string strInputName = strInputPath.substr(strInputPath.find_last_of("/") + 1);
		std::string strOutputPath = FLAGS_output_folder + "/result_" + strInputName;
		DrawRectangle(cvMatRgbImage, nObject);
		cv::imwrite(strOutputPath, cvMatRgbImage);
	}
  google::ShutdownGoogleLogging();
  return 0;
}