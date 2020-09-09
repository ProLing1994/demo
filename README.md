# demo

./common
1、3rd—party：gflags/glog
2、utils：file_system

./HelloWorld，hisi 3516D 板端代码
1、HelloWorld_himix200、Student_himix200 包含板端 makefile 文件
2、HelloWorld_linux、Student_linux 包含 Linux makfile 文件

./MNN，阿里 MNN 框架
1、test_mobilenet_ssd 测试 ssd 运行时间
2、test_mobilenet_ssd_thread 测试多线程 ssd 运动时间

./OpenVINO，intel 框架
1、test_mobilenet_ssd 测试 ssd 运行时间
2、test_mobilenet_ssd_thread 测试多线程 ssd 运动时间
3、test_mobilenet_ssd_yuv 输入 yuv 视频，进行车牌检测

./json_cpp，git clone https://github.com/ProLing1994/jsoncpp
该 demo 用于演示交叉编译。
1、编写 makeflie 文件，make/make clean。
2、针对 CmakeList.txt，编写 toolChain.cmake

./speech，kaldi 语音识别框架
1、online2-wav-nnet3-latgen-faster，对离线语音识别算法 chain model 进行时耗测试
