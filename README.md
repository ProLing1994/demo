# demo

./common����������
1��3rd��party��gflags/glog
2��utils/csrc��file_system
3��utils/python��pytorch ѵ���ű�����

./CrossCompilation������������
1��CrossCompilation/HelloWorld��hisi 3516D ��˴���
	1��HelloWorld_himix200��Student_himix200 ������� makefile �ļ�
	2��HelloWorld_linux��Student_linux ���� Linux makfile �ļ�
2��CrossCompilation/json_cpp��git clone https://github.com/ProLing1994/jsoncpp
�� demo ������ʾ������롣
	1����д makeflie �ļ���make/make clean��
	2����� CmakeList.txt����д toolChain.cmake

./MNN������ MNN ���
1��test_mobilenet_ssd ���� ssd ����ʱ��
2��test_mobilenet_ssd_thread ���Զ��߳� ssd �˶�ʱ��

./OpenVINO��intel ���
1��test_mobilenet_ssd ���� ssd ����ʱ��
2��test_mobilenet_ssd_thread ���Զ��߳� ssd �˶�ʱ��
3��test_mobilenet_ssd_yuv ���� yuv ��Ƶ�����г��Ƽ��

./Speech�������ű�
1��Speech/kaldi ����ʶ����
	1��online2-wav-nnet3-latgen-faster������������ʶ���㷨 chain model ����ʱ�Ĳ���
2��Speech/VAD ����������ű���Ŀǰʹ�����й��������ű�
3��Speech/KWS �����ؼ��ʼ����ű���pytorchѵ���ű�