import numpy as np
import os 
import sys
import torch
import torch.nn as nn

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe 


class RNNLM(nn.Module):
    def __init__(self, vocab_size,hidden_size, pretrained=True):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)       # 对字符编码, 不支持
        self.lstm=nn.LSTM(hidden_size, 512, 3, bias=True, dropout=0.3)
        self.fc_out=nn.Linear(512, vocab_size)
        self.softmax=nn.Softmax(dim=2)
        self.dropout=nn.Dropout(0.2)

    def forward(self, x,hidden=None):
        x = self.embedding(x)                           # shape: (batch, 32) -> shape: (batch, 32, 512)
        x = x.transpose(1,0)                            # shape: (batch, 32, 512) -> shape: (32, batch, 512)
        lstm_out, (hn, cn) = self.lstm(x,hidden)        # shape: (32, batch, 512) -> shape: (32, batch, 512)
        fc_out = self.fc_out(lstm_out)                  # shape: (32, batch, 512) -> shape: (32, batch, 408)
        return fc_out, (hn, cn), x

    def forward_onestep(self, x,hidden_pre):
        x = self.embedding(x)
        x = x.transpose(1, 0)
        lstm_out, (hn, cn) = self.lstm(x,hidden_pre)
        x = lstm_out[-1]
        fc_out = self.fc_out(x)
        score=self.softmax(fc_out)
        return score,(hn, cn)


if __name__ == '__main__':
    # input 
    model_path =  "/mnt/huanyuan/model/kws_model/RNN_language_model"
    pytorch_model = "cn_without_tone_lm_26.pth"
    caffe_prototxt = "cn_without_tone_lm_26.prototxt"
    caffe_model = "cn_without_tone_lm_26.caffemodel"

    # load pytorch model
    pytorch_network= RNNLM(408, 512)
    checkpoint = torch.load(os.path.join(model_path, pytorch_model), map_location=torch.device('cpu'))
    pytorch_network.load_state_dict(checkpoint['state_dict'])
    pytorch_network.eval()

    # pytorch forward
    input = torch.ones([1, 32]).long()
    fc_out, _, x = pytorch_network.forward(input)

    # load caffe prototxt

    caffe.set_mode_cpu()
    net = caffe.Net(os.path.join(model_path, caffe_prototxt), os.path.join(model_path, caffe_model), caffe.TEST)
    net.blobs['data'].reshape(1, 32, 1, 512) 
    net.blobs['data'].data[...] = np.expand_dims(x.data.numpy(), axis=0)
    
    net.blobs['clip'].reshape(1, 32, 1, 1) 
    clip_data = np.ones((32, 1, 1), dtype = np.int16)
    clip_data[0][0][0] = 0
    net.blobs['clip'].data[...] = np.expand_dims(clip_data, axis=0)
    net_output = net.forward()['fc_out'] 

    print("Pytorch output: \n", fc_out.data.numpy().reshape(net_output.shape))
    print("Caffe output: \n", net_output)