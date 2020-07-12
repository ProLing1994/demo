#include <iostream>
#include <memory>

#include "mobilenet_ssd_detector.h"
#include "opencv2/opencv.hpp"

int main(int argc, char* argv[]){
    std::cout << "hello" << std::endl;
    std::string model_path = "/home/huanyuan/code/models/mobilenet_iter_73000.caffe.mnn";
    std::string image_path = "/home/huanyuan/code/MobileNet-SSD/images/car.jpg";

    std::shared_ptr<MNN::MobilenetSSDDetector> mobilenet_ssd_detector;
    mobilenet_ssd_detector.reset(new MNN::MobilenetSSDDetector());
    int error_int = mobilenet_ssd_detector->init(model_path.c_str());

    cv::Mat img_src = cv::imread(image_path.c_str(), 1);
    std::vector<MNN::ObjectInformation> objects;
    mobilenet_ssd_detector->detect(img_src, &objects);

    int num_objects = static_cast<int>(objects.size());
		for (int i = 0; i < num_objects; ++i) {
			std::cout << "location: " << objects[i].location_ << std::endl;
			cv::rectangle(img_src, objects[i].location_, cv::Scalar(0, 0, 255), 2);
			char text[256];
			std::cout << objects[i].name_.c_str() << objects[i].score_ * 100 << std::endl;
			// sprintf_s(text, "%s %.1f%%", objects[i].name_.c_str(), objects[i].score_ * 100);
			// int baseLine = 0;
			// cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			// cv::putText(img_src, text, cv::Point(objects[i].location_.x,
			// 	objects[i].location_.y + label_size.height),
			// 	cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
        cv::imwrite("./car_result.jpg", img_src);
        // cv::imshow("image", img_src);
		// cv::waitKey(0);
    return 0;
}