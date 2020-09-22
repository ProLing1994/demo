#include "face_ssd_caffe_openvino.hpp"

#include <iostream>

#ifndef AIP_API
#define AIP_API __attribute__((visibility("default")))
#endif // AIP_API

AIP_API rmInferenceFaceSsdOpenvino::rmInferenceFaceSsdOpenvino(const OPENVINO_SSD_PARAM_S& params):
    mParams(params) {}

AIP_API rmInferenceFaceSsdOpenvino::~rmInferenceFaceSsdOpenvino() {}

AIP_API int rmInferenceFaceSsdOpenvino::build(){
  //!< 0.对配置文件进行核验
  if (mParams.u32ThreadNum <= 0 || mParams.u32ThreadNum > 32){
    std::cout << "ERROR, FaceRecOpenvino only support for 1~32, cannot be "<< mParams.u32ThreadNum << std::endl;
    return -1;
  }
  
  if (mParams.strModelPath.find(".xml") == mParams.strModelPath.npos) {
    std::cout << "ERROR, ModelPath type must be .xml" << std::endl;
    return -1;
	}

  //!< 1.加载推理引擎, load inference engine
  InferenceEngine::Core Ie;
  Ie.GetVersions("CPU"); //!< 设置CPU模式
  std::cout << "FaceSsdOpenvino use device Type: CPU" << std::endl;

  std::map<std::string, std::map<std::string, std::string>> nConfig;
  nConfig["CPU"] = {};
  std::map<std::string, std::string>& nDeviceConfig = nConfig.at("CPU");
  nDeviceConfig[CONFIG_KEY(CPU_THREADS_NUM)] = std::to_string(mParams.u32ThreadNum);
  for (auto&& item : nConfig) {
    Ie.SetConfig(item.second, item.first);
  }

  //!< 2.加载模型文件
	// std::string strBinFileName = mParams.strModelPath.substr(0, mParams.strModelPath.rfind('.')) + ".bin";
  m_Network = Ie.ReadNetwork(mParams.strModelPath);

  //!< 3.制作输入流, prepare input blobs
	InferenceEngine::InputsDataMap InputInfo(m_Network.getInputsInfo());
	if (InputInfo.size() != 1){
    std::cout << "ERROR, FaceSsdOpenvino has only one input but got " << InputInfo.size() << std::endl;
    return -1;
  }
	m_InputInfo = InputInfo.begin()->second;
  m_InputInfo->setPrecision(InferenceEngine::Precision::U8);
  m_InputInfo->setLayout(InferenceEngine::Layout::NCHW);
	m_strInputName = InputInfo.begin()->first;
    
  //!< 判断输入尺寸
  m_InputDims = m_InputInfo->getTensorDesc().getDims();
  if(m_InputDims.size() != 4){
    std::cout<<"ERROR, FaceSsdOpenvino input dim should be 4 but got "<< m_InputDims.size() << std::endl;
    return -1;
  } 
  if(m_InputDims[1] != 1 && m_InputDims[1] != 3){
    std::cout<< "ERROR, FaceSsdOpenvino input channels should be 1 or 3 but got " << m_InputDims[1] << std::endl;
    return -1;
  }

  //!< 4.制作输出流, prepare output blobs
	InferenceEngine::OutputsDataMap OutputInfo(m_Network.getOutputsInfo());
	if (OutputInfo.size() != 1){
    std::cout<< "ERROR, FaceSsdOpenvino has only one output but got "<< OutputInfo.size() << std::endl;
    return -1;
  }
  m_OutputInfo = OutputInfo.begin()->second;
  if (m_OutputInfo == nullptr){
    std::cout<< "ERROR, FaceSsdOpenvino do not have a output" << std::endl;
    return -1;
  }
  m_OutputInfo->setPrecision(InferenceEngine::Precision::FP32);
  m_strOutputName = m_OutputInfo->getName();
  
  //!< 判断输出特征维度
  m_OutputDims = m_OutputInfo->getTensorDesc().getDims();
	const size_t u32ObjectSize = m_OutputDims[3];
  if (m_OutputDims.size() != 4) {
    std::cout<<"ERROR, FaceSsdOpenvino output dim should be 4 but got "<< m_OutputDims.size() << std::endl;
	  return -1;
	}

	if (u32ObjectSize != 7) {
    std::cout<<"ERROR, FaceSsdOpenvino Output item should have 7 as a last dimension but got "<< m_OutputDims.size() << std::endl;
	  return -1;
	}
    
  //!< 5.加载模型至CPU, loading model to the device
  InferenceEngine::ExecutableNetwork ExecutableNetwork = Ie.LoadNetwork(m_Network, "CPU");

  //!< 6.创建推理上下文(位置待定)
	m_InferRrequest = ExecutableNetwork.CreateInferRequest();

  return 0;
}

AIP_API bool rmInferenceFaceSsdOpenvino::infer(const cv::Mat& cvMatImage, std::vector<OBJECT_INFO_S>* pstnObject){
  //!< 0.获取输入信息
  const unsigned int inputC = m_InputDims[1];
  const unsigned int inputH = m_InputDims[2];
  const unsigned int inputW = m_InputDims[3];
  const int batchSize = m_InputDims[0];

  //!< 1.输入尺寸归一化
  int s32ImageWidth = cvMatImage.cols;
	int s32ImageHeight = cvMatImage.rows;
  cv::Mat resize_ipt;
  cv::resize(cvMatImage, resize_ipt, cv::Size(inputW, inputH));
  if(inputC==1){
    cv::cvtColor(resize_ipt,resize_ipt,cv::COLOR_BGR2GRAY);
  }

  //!< 2.传入数据
  InferenceEngine::Blob::Ptr pImageIpt = m_InferRrequest.GetBlob(m_strInputName);
  // Filling input tensor with images. First b channel, then g and r channels 
  InferenceEngine::MemoryBlob::Ptr pstMemortImage = InferenceEngine::as<InferenceEngine::MemoryBlob>(pImageIpt);
  if (!pstMemortImage) {
    std::cout << "We expect image blob to be inherited from MemoryBlob, but by fact we were not able "
        "to cast imageInput to MemoryBlob" << std::endl;;
    return -1;
  }
  // locked memory holder should be alive all time while access to its buffer happens
  auto InputHolder = pstMemortImage->wmap();
  unsigned char *pstucData = InputHolder.as<unsigned char *>();
  /** Iterate over all pixel in image (b,g,r) **/
  for (size_t u32Pid = 0; u32Pid < inputH * inputW; u32Pid++) {
    /** Iterate over all channels **/
    for (size_t u32Ch = 0; u32Ch < inputC; ++u32Ch) {
      pstucData[u32Ch * inputH * inputW + u32Pid] = resize_ipt.data[u32Pid * inputC + u32Ch];
    }
  }

  //!< 3.do inference
  m_InferRrequest.Infer();

  //!< 4.处理输出结果
  const InferenceEngine::Blob::Ptr pstOutputBlob = m_InferRrequest.GetBlob(m_strOutputName);
  InferenceEngine::MemoryBlob::CPtr pstMemoryOutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(pstOutputBlob);
  if (!pstMemoryOutput) {
    std::cout << "We expect output to be inherited from MemoryBlob, "
    "but by fact we were not able to cast output to MemoryBlob"<<std::endl;;
    return -1;
  }

  // locked memory holder should be alive all time while access to its buffer happens
  auto OutputHolder = pstMemoryOutput->rmap();
  const float *pstFeatures = OutputHolder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

	const InferenceEngine::SizeVector nOutputDim = m_OutputInfo->getTensorDesc().getDims();
	const size_t u32MaxProposalCount = nOutputDim[2];
	const size_t u32ObjectSize = nOutputDim[3];

	/* Each detection has image_id that denotes processed image */
	for (int s32CurProposal = 0; s32CurProposal < u32MaxProposalCount; s32CurProposal++) {

	  float f32Confidence = pstFeatures[s32CurProposal * u32ObjectSize + 2];
	  auto label = static_cast<int>(pstFeatures[s32CurProposal * u32ObjectSize + 1]);
	  auto xmin = static_cast<int>(pstFeatures[s32CurProposal * u32ObjectSize + 3] * s32ImageWidth);
	  auto ymin = static_cast<int>(pstFeatures[s32CurProposal * u32ObjectSize + 4] * s32ImageHeight);
	  auto xmax = static_cast<int>(pstFeatures[s32CurProposal * u32ObjectSize + 5] * s32ImageWidth);
	  auto ymax = static_cast<int>(pstFeatures[s32CurProposal * u32ObjectSize + 6] * s32ImageHeight);
    xmin = std::max(0, xmin);
		xmin = std::min(xmin, s32ImageWidth - 1);
    ymin = std::max(0, ymin);	     
		ymin = std::min(ymin, s32ImageHeight - 1);
    xmax = std::max(0, xmax);
		xmax = std::min(xmax, s32ImageWidth - 1);
		ymax = std::max(0, ymax);	
		ymax = std::min(ymax, s32ImageHeight - 1);

	  if (f32Confidence > mParams.f64Threshold) {
		/** Drawing only objects with >50% probability **/
		OBJECT_INFO_S Object;
		Object.strClassName = mParams.nClassName[label];
		Object.f32Score = f32Confidence;
		Object.cvRectLocation.x = xmin;
		Object.cvRectLocation.y = ymin;
		Object.cvRectLocation.width = xmax - xmin;
		Object.cvRectLocation.height = ymax - ymin;
		pstnObject->push_back(Object);
	  }
	}
    return 0;
}