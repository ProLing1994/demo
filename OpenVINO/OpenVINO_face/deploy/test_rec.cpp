#include <iostream>
#include <time.h>
#include <string>

#include "opencv2/opencv.hpp"
#include "facerec_caffe_openvino.h"

int main(int argc, char** argv){
    std::string readpath = argv[1];
    cv::Mat img = cv::imread(readpath);

    OpenvinoRecParam recparams;
    recparams.u32ThreadNum = 4;
    recparams.strModelPath = "facerec-mask.xml";

    rmInferenceFaceRecOpenvino* gREC;
    gREC = new rmInferenceFaceRecOpenvino(recparams);

    gREC->build();

    std::vector<float> features;
    gREC->infer(img, features);

    for(int j=0;j<features.size();j++){
        std::cout<<features[j]<<", ";
    }

    return 0;
    // SampleRecParams recparams;
    // recparams.prototxtFileName = pRECproto;
    // recparams.weightsFileName = pRECmodel;
    // recparams.inputTensorNames.push_back("data");
    // recparams.batchSize = 1;
    // recparams.outputTensorNames.push_back("batch_norm_blob57");
    // recparams.outputDim = 512;
    // recparams.pixelMean[0] = 127.5f;recparams.pixelMean[1] = 127.5f;recparams.pixelMean[2] = 127.5f;
    // recparams.pixelScale = 0.0078431;

}