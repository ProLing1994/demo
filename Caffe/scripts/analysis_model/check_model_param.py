import sys
caffe_root = "/home/huanyuan/code/caffe/"
sys.path.insert(0, caffe_root+'python')
import caffe

if __name__ == "__main__":
    deploy = "/home/huanyuan/code/MNN/models/face_no_bn.prototxt"
    caffe_model =  "/home/huanyuan/code/MNN/models/face_no_bn.caffemodel"

    net = caffe.Net(deploy, caffe_model, caffe.TEST)

    conv1_w = net.params['p3_1'][0].data
    print(conv1_w.shape)
    conv1_w_reshape = conv1_w.reshape((conv1_w.shape[0] * conv1_w.shape[1] * conv1_w.shape[2] * conv1_w.shape[3]))
    print(conv1_w_reshape[:10])
    conv1_b = net.params['p3_1'][1].data
    print(conv1_b[:10])


