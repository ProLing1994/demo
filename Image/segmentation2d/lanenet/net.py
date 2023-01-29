import os
import cv2
import copy
import torch
import numpy as np
# from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from time import time
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import lr_scheduler

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loss.FocalLoss import FocalLoss2d
from loss.segloss import *
from loss.RMILoss import RMI_Topk_loss
from models.backbone import *
from models.head import HeadCNN
from models.LANENET import LANENET_NOVT, LANENET_BIG
# from models.segformer_pytorch import Segformer
# from models.swinunet.swin_transformer_unet import SwinTransformerSys
# from models.swin.buildnet import swin_seg
# from models.crossformer_backbone import crossformer
# from models.setr import SETR
from utils.Metric import SegMetric
from utils.utils import OPcounter, trainval_split, writelist
from utils.utils import buildDir, readlist, rebuildDir, get_filelist
from utils.utils_mask import * 
from dataset.dataset_default import Segdataset
from GetDataList import read_weight_list # get class_weight

# from cfgs.cfg_base import *
from cfgs.cfg_schoolbus import *

class BuildNet(nn.Module):
    def __init__(self, backbone, nclass, img_c, img_hw, fc, type_block, nblock, 
                 downsample_mode='conv', upsample_mode='conv', softmax=False):
        super(BuildNet, self).__init__()
        if backbone == 'Seg16':
            net = Seg16
        elif backbone == 'Fpn16':
            net = Fpn16
        elif backbone == 'FCN16':
            net = FCN16
        elif backbone == 'mix16':
            net = mix16
        elif backbone == 'Seg8':
            net = Seg8
        
        self.bkb  = net(img_c, img_hw, fc, type_block, nblock, downsample_mode, upsample_mode)
        self.head = HeadCNN(fc, nclass, softmax)
    def forward(self, x):
        x = self.bkb(x)
        x = self.head(x)
        # x = F.sigmoid(x)
        return x

class SegNet(object):
    def __init__(self, args):
        # dirs
        self.args = args
        self.get_datalist2(DATALIST)
        self.modelcode = '{}_{}'.format(cfg_modelcode, cfg_modeltype)
        self.logdir = '{}/{}/{}'.format(args.rootdir, args.logdir, self.modelcode)
        self.weightdir = '{}/{}/{}'.format(args.rootdir, args.weightdir, self.modelcode)
        self.results = '{}/{}/outputs/{}'.format(args.rootdir, args.results, self.modelcode)
        print(self.modelcode)

        # args
        self.lr = args.lr
        self.loss = args.loss
        self.phase = args.phase
        self.resume = args.resume
        self.device = args.device
        self.augrate = args.augrate
        self.workers = args.workers
        self.lr_decay = args.lr_decay
        self.lr_gamma = args.lr_gamma
        self.batchsize = args.batchsize
        self.max_epoch = args.max_epoch
        self.labelmode = args.labelmode
        self.labelfile = args.labelfile
        self.onlinetest = args.onlinetest
        self.img_c = cfg_img_c
        self.nclass = len(ClassTable)
        self.backbone = cfg_backbone
        self.input_shape = cfg_img_hw
        self.rgb_means = cfg_rgb_means
        self.class_table = ClassTable
        self.colormap = np.asarray(cfg_color_map)
        self.fullmap = True
        self.class_weight = torch.Tensor(cfg_class_weight).to(self.device) # self.get_class_weight(cfg['class_weight'])
        
        self.modeltype = cfg_modeltype
        
        # model
        if self.modeltype == 'huge':
            # type 1
            fc              = args.fc
            nblock          = args.nblock
            downsample_mode = args.downsample_mode
            upsample_mode   = args.upsample_mode
            type_block      = cfg_type_block
            self.net = BuildNet(self.backbone, self.nclass, self.img_c, self.input_shape, fc, 
                                type_block, nblock, downsample_mode, upsample_mode, softmax=False)
        elif self.modeltype == 'lanenet':
            # type 2
            model_init_nclass = self.nclass if self.labelmode=='single' else self.nclass-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = LANENET_NOVT(num_classes=model_init_nclass, sigmoid=do_sigmoid)
        elif self.modeltype == 'lanenet_big':
            model_init_nclass = self.nclass if self.labelmode=='single' else self.nclass-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = LANENET_BIG(num_classes=model_init_nclass, sigmoid=do_sigmoid)
        """
        elif self.modeltype == 'segformer':
            # type 3
            model_init_nclass = self.nclass if self.labelmode=='single' else self.nclass-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = Segformer(num_classes=model_init_nclass, sigmoid=do_sigmoid)
        elif self.modeltype == 'swinunet':
            model_init_nclass = self.nclass if self.labelmode=='single' else self.nclass-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = SwinTransformerSys(img_size=self.input_shape[0], num_classes=model_init_nclass, 
                                          window_size=8, do_sigmoid=do_sigmoid)
        elif self.modeltype == 'swin':
            model_init_nclass = self.nclass if self.labelmode=='single' else self.nclass-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = swin_seg(self.input_shape[0], model_init_nclass, do_sigmoid)
        elif self.modeltype == 'crossformer':
            model_init_nclass = self.nclass if self.labelmode=='single' else self.nclass-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = crossformer(self.input_shape[0], model_init_nclass, do_sigmoid)
        elif self.modeltype == 'setr':
            model_init_nclass = self.nclass if self.labelmode=='single' else self.nclass-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = SETR.SETR_PUP(
                img_dim=self.input_shape[0], 
                patch_dim=8, 
                num_channels=3, 
                num_classes=model_init_nclass, 
                embedding_dim=768, 
                num_heads=12, 
                num_layers=8, 
                hidden_dim=3072,
                dropout_rate=0.1, 
                attn_dropout_rate=0.1,  
                do_sigmoid = do_sigmoid
                )
        else:
            raise ValueError
        """
        self.net = self.net.to(self.device)
        self.ShowOpInfo()
        
        # component
        if self.labelmode == 'single':
            if self.loss == 'ce':
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weight).to(self.device)
            elif self.loss == 'focal':
                self.criterion = FocalLoss2d(gamma=2, weight=self.class_weight).to(self.device)
            elif self.loss == 'dcce':
                self.criterion = DC_CE_loss(class_weight=self.class_weight).to(self.device)
            elif self.loss == 'dcfc':
                self.criterion = DC_Focal_loss()
            elif self.loss == 'gdlce':
                self.criterion = GDL_CE_loss()
            elif self.loss == 'dcfcce':
                self.criterion = DC_FC_CE(class_weight=self.class_weight, weight_ce=1, weight_fc=1.5, weight_dice=1.0)
            elif self.loss == 'rmitopk':
                self.criterion = RMI_Topk_loss(seg_num_classes=self.nclass, class_weight=self.class_weight)
        else:
            self.criterion = nn.BCELoss().to(self.device)
        
        self.optimizer = optim.Adam(self.net.parameters(), self.lr, amsgrad=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay, gamma=self.lr_gamma)
        
        # data
        self.dataset = Segdataset(self.trainlist, self.class_table, self.input_shape, mode='train', aug_rate=self.augrate, rgb_means=self.rgb_means)
        self.testset = Segdataset(self.testlist , self.class_table, self.input_shape, mode='test' , aug_rate=self.augrate, rgb_means=self.rgb_means)
        self.dataloader = data.DataLoader(self.dataset, self.batchsize, shuffle=True, num_workers=self.workers, pin_memory=True)
        self.testloader = data.DataLoader(self.testset, self.batchsize, shuffle=True, num_workers=self.workers, pin_memory=True)
        print('*** Length of trainset: {} ***'.format(len(self.dataset)))
        print('*** Length of testset: {} ***'.format(len(self.testset)))
        print('*** Class Weights: {} ***'.format([round(float(i), 3) for i in self.class_weight]))
        
        # log
        self.train_log = {
            'train_loss_epoch':[], 
            'test_loss_epoch':[], 
            'train_acc_epoch':[], 
            'test_acc_epoch':[], 
            'train_loss_iter':[],
            'train_acc_iter':[], 
        }
        
        self.metrics_log_txt_path = '{}/metrics_log.txt'.format(self.logdir)
        if os.path.exists(self.metrics_log_txt_path):
            self.metrics_log_txt = readlist(self.metrics_log_txt_path)
        else:
            head = 'epoch, loss, acc, PA, mPA, mIoU, '
            for i in range(self.nclass-1):
                head += 'cPA-{}, '.format(self.class_table[i+1])
            for i in range(self.nclass-1):
                head += 'cIoU-{}, '.format(self.class_table[i+1])
            self.metrics_log_txt = [head]
        
        if self.onlinetest:
            if self.labelmode == 'single':
                logpath = '{}/metrics_log.npy'.format(self.logdir)
                if os.path.exists(logpath):
                    self.metrics_log = np.load(logpath, allow_pickle=True).item()
                else:
                    self.metrics_log = {
                        'PA':[], 
                        'mPA':[], 
                        'mIoU':[], 
                        'cPA':[], 
                        'cIoU':[], 
                    }
            else:
                self.metrics_log = []
                for c in range(1, self.nclass):
                    logpath = '{}/metrics_log_{}.npy'.format(self.logdir, self.class_table[c])
                    if os.path.exists(logpath):
                        metrics_log_c = np.load(logpath, allow_pickle=True).item()
                    else:
                        metrics_log_c = {
                            'PA':[], 
                            'mPA':[], 
                            'mIoU':[], 
                            }
                    self.metrics_log.append(metrics_log_c)

    def train_net(self):
        buildDir(self.weightdir)
        buildDir(self.logdir)
        
        print('*** SegNet training start! ***')
        self.phase == 'train'
        self.load_weights()
        self.get_start_epoch()
        self.get_start_lr()
        self.get_min_loss()
        self.get_max_acc()

        for epoch in range(self.start_epoch, self.max_epoch):
            # train
            self.net.train()
            torch.set_grad_enabled(True)
            tot_acc = 0
            tot_loss = 0
            tot_time = 0
            self.scheduler.step()
            train_tot_t1 = time()
            for batch, (images, targets) in enumerate(self.dataloader):
                bsize = len(images)
                images  = Variable(images.to(self.device))
                targets = Variable(targets.to(self.device)) # single:b,h,w / multi:b,nclass-1,h,w
                # targets = targets*1.0 if self.labelmode == 'multi' else targets[:, 0, ...]
                t_start = time()
                out = self.net(images) # single:b,nclass,h,w / multi:b,nclass-1,h,w
                self.optimizer.zero_grad()
                if self.loss in ['dcce', 'dcfc', 'gdlce', 'dcfcce'] and self.labelmode=='single':
                    loss = self.criterion(out, torch.unsqueeze(targets, 1))
                else:
                    loss = self.criterion(out, targets)
                loss.backward()
                self.optimizer.step()
                t_end = time()
                tot_loss += bsize * loss.data.item()
                acc = self.count_acc(out, targets, self.labelmode)
                tot_acc += acc * bsize
                tot_time += t_end - t_start
                if batch == 0 or (batch+1) % 50 == 0:
                    print('[epoch {}/{}] [iter {}/{}] loss: {:.5f} acc: {:.5f} lr: {:.5f}'\
                        .format(epoch+1, self.max_epoch, batch+1, len(self.dataloader), loss.data.item(), acc, self.scheduler.get_lr()[0]))
                # updata per iter
                self.train_log['train_loss_iter'].append(loss.data.item())
                self.train_log['train_acc_iter'].append(acc)
            tot_loss /= len(self.dataset)
            tot_acc /= len(self.dataset)
            tot_time /= len(self.dataset)
            train_tot_t2 = time()
            self.train_log['train_loss_epoch'].append(tot_loss)
            self.train_log['train_acc_epoch'].append(tot_acc)
            print('[epoch {}/{}] train_loss: {:.5f} train_acc: {:.5f} sec/iter: {:.3f}s tot: {:.1f}min'\
                .format(epoch+1, self.max_epoch, tot_loss, tot_acc, tot_time, (train_tot_t2-train_tot_t1)/60))
            
            # validation / test
            self.net.eval()
            torch.set_grad_enabled(False)
            tot_loss = 0
            tot_acc = 0
            for batch, (images, targets) in enumerate(self.testloader):
                bsize = len(images)
                images  = Variable(images.to(self.device))
                targets = Variable(targets.to(self.device)) # single:b,h,w / multi:b,nclass-1,h,w
                # targets = targets*1.0 if self.labelmode == 'multi' else targets[:, 0, ...]
                out = self.net(images) # single:b,nclass,h,w / multi:b,nclass-1,h,w
                if self.loss in ['dcce', 'dcfc', 'gdlce', 'dcfcce'] and self.labelmode=='single':
                    loss = self.criterion(out, torch.unsqueeze(targets, 1))
                else:
                    loss = self.criterion(out, targets)
                tot_loss += bsize * loss.data.item()
                tot_acc += self.count_acc(out, targets, self.labelmode) * bsize
            
            tot_loss /= len(self.testset)
            tot_acc /= len(self.testset)
            save_flag = self.save_weights(epoch+1, tot_loss, tot_acc)
            print('[epoch {}/{}] eval_loss : {:.5f} eval_acc : {:.5f} --{}'\
                .format(epoch+1, self.max_epoch, tot_loss, tot_acc, save_flag))
            self.train_log['test_loss_epoch'].append(tot_loss)
            self.train_log['test_acc_epoch'].append(tot_acc)
            
            if self.onlinetest:
                self.test_net(epoch+1, True)
            
            self.updata_curve()
            print('\n')
        print('*** Training is complete, max acc: {:.3f}% ***'.format(float(self.max_acc)*100))

    def test_net(self, epoch, online=False):
        # online or offline (after training)
        # single or multi (label)
        self.phase = 'test'
        if online:
            pass
        else:
            self.load_weights(epoch)
        
        self.net.eval()
        torch.set_grad_enabled(False)
        if self.labelmode == 'single':
            metrics = SegMetric(self.nclass, self.device)
        elif self.labelmode == 'multi':
            metrics = []
            for _ in range(self.nclass-1):
                metrics.append(SegMetric(2, self.device))
        print('*** testing start! ***')
        
        tot_loss = 0
        tot_acc = 0
        for batch, (images, targets) in tqdm(enumerate(self.testloader), total=len(self.testloader)):
            bsize = len(images)
            images  = Variable(images.to(self.device))
            targets = Variable(targets.to(self.device))
            # targets = targets*1.0 if self.labelmode == 'multi' else targets[:, 0, ...]
            out = self.net(images) # single:bchw / multi:(b,nclass-1,h,w)
            if self.loss in ['dcce', 'dcfc', 'gdlce', 'dcfcce'] and self.labelmode=='single':
                loss = self.criterion(out, torch.unsqueeze(targets, 1))
            else:
                loss = self.criterion(out, targets)
            tot_loss += bsize * loss.data.item()
            tot_acc += self.count_acc(out, targets, self.labelmode) * bsize
            
            targets = targets.long()
            if self.labelmode == 'single':
                pred = torch.argmax(F.softmax(out, 1), 1) # single: bhw
                metrics.addBatch(pred, targets)
            elif self.labelmode == 'multi':
                pred = out.round() # multi: (b,nclass-1,h,w)
                pred = pred.long()
                
                for c in range(self.nclass - 1):
                    pred_c = pred[:, c, ...]
                    target_c = targets[:, c, ...]
                    metrics[c].addBatch(pred_c, target_c)
        
        tot_loss /= len(self.testset)
        tot_acc /= len(self.testset)
        tot_loss = float(tot_loss)
        tot_acc = float(tot_acc)
        
        if self.labelmode == 'single':
            PA = metrics.pixelAccuracy()
            mPA = metrics.meanPixelAccuracy()
            cPA = metrics.classPixelAccuracy().cpu().numpy()
            mIoU = metrics.meanIntersectionOverUnion()
            cIoU = metrics.classIOU().cpu().numpy()
            
            if online:
                self.metrics_log['PA'].append(PA)
                self.metrics_log['mPA'].append(mPA)
                self.metrics_log['mIoU'].append(mIoU)
                self.metrics_log['cPA'].append(cPA)
                self.metrics_log['cIoU'].append(cIoU)
                for i in range(self.nclass - 1):
                    print('cPA-{} : {:.3f}'.format(self.class_table[i+1], cPA[i+1]))
                for i in range(self.nclass - 1):
                    print('cIoU-{}: {:.3f}'.format(self.class_table[i+1], cIoU[i+1]))
                
                line  = '{:^5}, '.format(epoch)
                line += '{:.5f}, '.format(tot_loss)
                line += '{:.5f}, '.format(tot_acc)
                line += '{:.5f}, '.format(PA)
                line += '{:.5f}, '.format(mPA)
                line += '{:.5f}, '.format(mIoU)
                for i in range(self.nclass - 1):
                    line += '{:.5f}, '.format(cPA[i+1])
                for i in range(self.nclass - 1):
                    line += '{:.5f}, '.format(cIoU[i+1])
                self.metrics_log_txt.append(line)
                writelist(self.metrics_log_txt, self.metrics_log_txt_path)
            else:
                infos = []
                infos.append('loss : {:.5f}'.format(tot_loss))
                infos.append('acc  : {:.5f}'.format(tot_acc))
                infos.append('PA   : {:.5f}'.format(PA))
                infos.append('mPA  : {:.5f}'.format(mPA))
                infos.append('cPA  : {}'.format(cPA))
                infos.append('mIoU : {:.5f}'.format(mIoU))
                infos.append('cIoU : {}'.format(cIoU))
                
                filepath = '{}/TestRecord_epoch{}.txt'.format(self.logdir, epoch)
                with open(filepath, 'w+') as f:
                    print('*** test scores: {} ***'.format(epoch))
                    for v in infos:
                        print('*** {} ***'.format(v))
                        f.write(v + '\n')
                print('*** test record saved in logs. ***')
                
        elif self.labelmode == 'multi':
            infos = []
            infos.append('tot loss : {:.5f}'.format(tot_loss))
            infos.append('tot acc  : {:.5f}'.format(tot_acc))
            for c in range(self.nclass - 1):
                PA = metrics[c].pixelAccuracy()
                mPA = metrics[c].meanPixelAccuracy()
                cPA = metrics[c].classPixelAccuracy().cpu().numpy()
                mIoU = metrics[c].meanIntersectionOverUnion()
                if online:
                    self.metrics_log[c]['PA'].append(PA)
                    self.metrics_log[c]['mPA'].append(mPA)
                    self.metrics_log[c]['mIoU'].append(mIoU)
                infos.append('class: {}'.format(self.class_table[c+1]))
                infos.append('PA   : {:.5f}'.format(PA))
                infos.append('mPA  : {:.5f}'.format(mPA))
                infos.append('mIoU : {:.5f}'.format(mIoU))
            
            if online:
                pass
                # plot figs : updata_curve
                # save logs : updata_curve
            else:
                filepath = '{}/TestRecord_epoch{}.txt'.format(self.logdir, epoch)
                with open(filepath, 'w+') as f:
                    print('*** test scores: {} ***'.format(epoch))
                    for v in infos:
                        print('*** {} ***'.format(v))
                        f.write(v + '\n')
                print('*** test record saved in logs. ***')

    def deal(self, img):
        img_ipt = self.img_preprocess(img)
        output = self.net(img_ipt)
        maskimg = self.put_info(img, output, self.labelmode)
        return maskimg

    def inference(self, srcdir, mode='view', savedir=None, epoch=None):
        # mode: view/save
        if mode == 'save':
            buildDir(savedir)
        
        self.phase = 'inference'
        self.net.eval()
        torch.set_grad_enabled(False)
        self.load_weights(epoch)
        
        datalist = os.listdir(srcdir)
        datalist = [i for i in datalist if i.endswith('jpg') or i.endswith('png')]
        if len(datalist) == 0:
            print('inference dir is empty, please check!')
        
        print('*** Found {} imgs, inference start! ***'.format(len(datalist)))
        for k in tqdm(range(len(datalist))):
            imgpath = '{}/{}'.format(srcdir, datalist[k])
            img_ori = cv2.imread(imgpath)
            img_ipt = self.img_preprocess(img_ori)
            output = self.net(img_ipt)
            maskimg = self.put_info(img_ori, output, self.labelmode)
            if mode == 'save':
                cv2.imwrite('{}/{}'.format(savedir, datalist[k]), maskimg)
            elif mode == 'view':
                cv2.imshow('', maskimg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                raise NameError(mode)

    def inference_video(self, videodir, mode='view', savedir=None, epoch=None, jump=5, roi=None, wpath=None):
        if mode == 'save':
            assert savedir is not None
            buildDir(savedir)
        self.phase = 'inference'
        self.net.eval()
        torch.set_grad_enabled(False)
        if wpath is None:
            self.load_weights(epoch)
        else:
            self.net.load_state_dict(torch.load(wpath, map_location='cuda:0'))
        
        videolist = os.listdir(videodir)
        videolist = [i for i in videolist if i.split('.')[-1].lower() in ['mp4', 'avi', 'avi', 'rmvb', 'mov']]
        if len(videolist) == 0:
            print('inference dir is empty, please check!')
        
        print('*** Found {} videos, inference start! ***'.format(len(videolist)))
        for k in range(len(videolist)):
            print('processing No.{} video, total: {}.'.format(k+1, len(videolist)))
            vidpath = '{}/{}'.format(videodir, videolist[k])
            savepath = '{}/{}'.format(savedir, videolist[k].replace('.avi', '.mp4'))
            reader = cv2.VideoCapture(vidpath)
            num, fps = int(reader.get(7)), int(reader.get(5))
            hei, wid = int(reader.get(4)), int(reader.get(3))
            if mode == 'save':
                writer = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'XVID'), fps, (wid, hei))
            for f in tqdm(range(num)):
                (flag, img_ori) = reader.read()
                if not flag:
                    break
                if roi is not None:
                    x1,y1,x2,y2 = roi
                    img_ipt = img_ori[y1:y2,x1:x2, :]
                else:
                    img_ipt = img_ori
                img_res = self.deal(img_ipt)
                if roi is not None:
                    img_show = img_ori.copy()
                    img_show[y1:y2,x1:x2, :] = img_res
                    img_show = cv2.rectangle(img_show, (x1,y1), (x2,y2), (255,255,255), 3)
                else:
                    img_show = img_res
                if mode == 'save':
                    writer.write(img_show)
                else:
                    cv2.imshow('', img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                
                for _ in range(jump):
                    reader.read()
                    f += jump
            if mode == 'save':
                writer.release()
            reader.release()
    
    def save_weights(self, step, loss, acc, mode='acc'):
        code = 'Seg'
        # save weights every epoch
        torch.save(self.net.state_dict(), "{}/{}_epoch{}.pth".format(self.weightdir, code, step), _use_new_zipfile_serialization=False)
        # save best weights
        save_flag = 'pass'
        if mode == 'loss' and loss <= self.min_loss:
            self.min_loss = loss
            torch.save(self.net.state_dict(), "{}/{}_best.pth".format(self.weightdir, code), _use_new_zipfile_serialization=False)
            save_flag = 'save'
        if mode == 'acc' and acc >= self.max_acc:
            self.max_acc = acc
            torch.save(self.net.state_dict(), "{}/{}_best.pth".format(self.weightdir, code), _use_new_zipfile_serialization=False)
            save_flag = 'save'
        return save_flag
        
    def load_weights(self, epoch=None):
        code = 'Seg'
        if self.phase == 'train':
            if self.resume != '':
                try:
                    self.net.load_state_dict(torch.load('{}/{}_{}.pth'.format(self.weightdir, code, self.resume), self.device))
                    print('*** Loading Weights Succeeded! ***')
                except:
                    try:
                        self.net.state_dict().update(torch.load('{}/{}_best.pth'.format(self.weightdir, code, self.resume), self.device).items())
                        print('*** Updating Weights Succeeded! ***')
                    except:
                        print('*** Train From Begining! ***')
            else:
                print('*** Train From Begining! ***')
        elif self.phase in ['test', 'inference']:
            if epoch is None or epoch=='best':
                self.net.load_state_dict(torch.load('{}/{}_best.pth'.format(self.weightdir, code), self.device))
            else:
                self.net.load_state_dict(torch.load('{}/{}_epoch{}.pth'.format(self.weightdir, code, str(epoch)), self.device))
            print('*** Loading Weights Succeeded! ***')

    def updata_curve(self):
        log = self.train_log
        np.save('{}/train_log'.format(self.logdir), log)
        keys = list(log.keys())
        
        plt.figure()
        plt.plot(log[keys[-1]], 'r')
        plt.xlabel('iter')
        plt.ylabel('acc')
        plt.title('acc per iter')
        plt.savefig('{}/acc_per_iter.png'.format(self.logdir), dpi=300)
        plt.close()
    
        plt.figure()
        plt.plot(log[keys[-2]], 'r')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.title('loss per iter')
        plt.savefig('{}/loss_per_iter.png'.format(self.logdir), dpi=300)
        plt.close()

        # train/test_loss_epoch
        c1 = log['train_loss_epoch']
        c2 = log['test_loss_epoch']
        if len(c1)>0 and len(c2)>0:
            idx = np.argmin(c2)
            val = c2[idx]
            plt.figure()
            plt.plot(c1, 'r')
            plt.plot(c2, 'b')
            plt.plot(idx, val, 'r.')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train loss', 'test loss'])
            plt.title('loss per epoch (best: {})'.format(idx+1))
            plt.savefig('{}/loss_per_epoch.png'.format(self.logdir), dpi=300)
            plt.close()
        
        c1 = log['train_acc_epoch']
        c2 = log['test_acc_epoch']
        if len(c1)>0 and len(c2)>0:
            idx = np.argmax(c2)
            val = c2[idx]
        plt.figure()
        plt.plot(c1, 'r')
        plt.plot(c2, 'b')
        plt.plot(idx, val, 'r.')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(['train acc', 'test acc'])
        plt.title('acc per epoch (best: {})'.format(idx+1))
        plt.savefig('{}/acc_per_epoch.png'.format(self.logdir), dpi=300)
        plt.close()
        
        if self.onlinetest:
            if self.labelmode == 'single':
                log = self.metrics_log
                
                cpa = log['cPA']
                ciou = log['cIoU']
                length = len(cpa)
                cpa_mat = np.zeros((length, self.nclass-1))
                ciou_mat = np.zeros((length, self.nclass-1))
                for i in range(length):
                    cpa_mat[i, :] = np.array(cpa[i][1:])
                    ciou_mat[i, :] = np.array(ciou[i][1:])
                
                np.save('{}/metrics_log'.format(self.logdir), log)
                plt.figure()
                plt.plot(log['PA'], 'r')
                plt.plot(log['mPA'], 'b')
                plt.plot(log['mIoU'], 'g')
                for i in range(self.nclass-1):
                    plt.plot(cpa_mat[:, i], '.-')
                for i in range(self.nclass-1):
                    plt.plot(ciou_mat[:, i], '*-')
                
                plt.xlabel('epoch')
                plt.ylabel('metrics')
                plt.legend(['PA', 'mPA', 'mIoU'] 
                           + ['cPA-{}'.format(i) for i in self.class_table[1:]] 
                           + ['cIoU-{}'.format(i) for i in self.class_table[1:]])
                plt.title('metrics per epoch')
                plt.savefig('{}/metrics_per_epoch.png'.format(self.logdir), dpi=300)
                plt.close()

            else:
                for c in range(self.nclass - 1):
                    log = self.metrics_log[c]
                    np.save('{}/metrics_log_{}'.format(self.logdir, self.class_table[c+1]), log)
                    plt.figure()
                    plt.plot(log['PA'], 'r')
                    plt.plot(log['mPA'], 'b')
                    plt.plot(log['mIoU'], 'g')
                    plt.xlabel('epoch')
                    plt.ylabel('metrics')
                    plt.legend(['PA', 'mPA', 'mIoU'])
                    plt.title('metrics per epoch {}'.format(self.class_table[c+1]))
                    plt.savefig('{}/metrics_per_epoch_{}.png'.format(self.logdir, self.class_table[c+1]), dpi=300)
                    plt.close()

    def get_start_epoch(self):
        if self.resume:
            try:
                log = np.load('{}/train_log.npy'.format(self.logdir), allow_pickle=True).item()
                train_loss_epoch = log['train_loss_epoch']
                start_epoch = len(train_loss_epoch)
                self.train_log = log
                print('load training log succeed!')
            except:
                print('training log not found, please check!')
                start_epoch = 0
        else:
            start_epoch = 0
        print('start epoch: {}'.format(start_epoch+1))
        self.start_epoch = start_epoch
    
    def get_start_lr(self):
        self.scheduler.step()
        for i in range(self.start_epoch):
            self.scheduler.step()
        print('start lr: {:.5f}'.format(self.scheduler.get_lr()[0]))
    
    def get_min_loss(self):
        if self.start_epoch:
            self.min_loss = np.min(self.train_log['test_loss_epoch'])
            print('min loss of history: {:.5f}'.format(self.min_loss))
        else:
            self.min_loss = np.float('inf')
            print('init test loss: {:.5f}'.format(self.min_loss))
    
    def get_max_acc(self):
        if self.start_epoch:
            self.max_acc = np.max(self.train_log['test_acc_epoch'])
            print('max acc  of history: {:.5f}'.format(self.max_acc))
        else:
            self.max_acc = 0
            print('init test acc : {:.5f}'.format(self.max_acc))
    
    def get_class_weight(self, mode):
        if mode == 'auto':
            try:
                class_weight = read_weight_list(self.datadir)
                class_weight = torch.Tensor(class_weight)
            except:
                mode = 'ones'
        if mode == 'ones':
            class_weight = torch.Tensor(torch.ones(self.nclass))
        return class_weight
    
    def img_preprocess(self, img):
        img = cv2.resize(img, self.input_shape[::-1])
        img = img[:,:,::-1] # bgr2rgb
        img = img.astype(np.float32)
        img -= self.rgb_means
        img = img.transpose((2,0,1))
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        img = Variable(img.to(self.device))
        return img

    def put_info(self, img, output, mode='single'):
        # mode: single/multi
        hi, wi, _ = img.shape
        if mode == 'single':
            result = torch.argmax(output, dim=1).cpu().numpy()
            result = result[0] # (h,w)
            rgbmask = mask2rgb(result, self.colormap)
            hm, wm, _ = rgbmask.shape
            if hi!= hm or wi!=wm:
                rgbmask = cv2.resize(rgbmask, (wi, hi), cv2.INTER_NEAREST)
            maskimg = cv2.addWeighted(img, 0.9, rgbmask, 0.5, 0)
        elif mode == 'multi':
            output = output[0, ...].cpu().numpy() # after sigmoid
            output = np.transpose(output, (1,2,0))
            results = np.rint(output).astype(int)
            hm, wm, _ = results.shape
            masklist = multi_mask2rgb(results, self.nclass-1, self.colormap[1:])
            maskimg = img.copy()
            for i in range(self.nclass-1):
                mask = masklist[i]
                if hi!= hm or wi!=wm:
                    mask = cv2.resize(mask, (wi, hi), cv2.INTER_NEAREST)
                maskimg = cv2.addWeighted(maskimg, 0.8, mask, 0.5, 0)
        return maskimg

    def ShowOpInfo(self):
        net = copy.deepcopy(self.net)
        flops, params = OPcounter(net, 3, self.input_shape, self.device, False)
        print('*** SEG flops: {} params: {} ***'.format(flops, params))

    def count_acc(self, out, target, mode='single'):
        if mode == 'single':
            # out: (b, c, h, w)
            # tar: (b,    h, w)
            b = target.shape[0]
            h, w = target.shape[-2:]
            pred = torch.argmax(F.softmax(out, 1), 1)
            count = torch.sum(pred==target) #if len(target.shape)==3 else torch.sum(pred==target[:, 0, ...])
            return float(count)/float(b*h*w)
        elif mode == 'multi':
            # out: (b, nclass, h, w)
            # tar: (b, nclass, h, w)
            out = out.round()
            return float (torch.sum(out==target) / target.numel())

    def get_datalist(self, datalist):
        train_data = []
        test_data = []
        for i in range(len(datalist)):
            tmp_train_data = readlist('{}/datalist/BeltTrainLoc.txt'.format(datalist[i]))
            tmp_test_data = readlist('{}/datalist/BeltTestLoc.txt'.format(datalist[i]))
            
            tmp_train_data = ['{}/Images/{}.jpg, {}/Annotations/{}.json, {}, {}'.format(datalist[i], x.split(', ')[0], datalist[i], x.split(', ')[0], x.split(', ')[1], x.split(', ')[2]) for x in tmp_train_data]
            tmp_test_data = ['{}/Images/{}.jpg, {}/Annotations/{}.json, {}, {}'.format(datalist[i], x.split(', ')[0], datalist[i], x.split(', ')[0], x.split(', ')[1], x.split(', ')[2]) for x in tmp_test_data]
            
            train_data += tmp_train_data
            test_data += tmp_test_data
            
        self.trainlist = train_data
        self.testlist = test_data
        return self.trainlist, self.testlist

    def get_datalist2(self, datalist):
        trainlist = []
        testlist = []
        for droot in datalist:
            train_datadir = '{}/train'.format(droot)
            test_datadir = '{}/test'.format(droot)

            train_datalist = get_filelist(train_datadir, ['.jpg'])
            test_datalist = get_filelist(test_datadir, ['.jpg'])

            train_id_list = [i.split('/')[-1].split('.jpg')[0] for i in train_datalist]
            test_id_list = [i.split('/')[-1].split('.jpg')[0] for i in test_datalist]
            
            _trainlist = ['{}/{}.jpg, {}/{}.json'.format(train_datadir, i, train_datadir, i) for i in train_id_list]
            _testlist = ['{}/{}.jpg, {}/{}.json'.format(test_datadir, i, test_datadir, i) for i in test_id_list]
            
            trainlist += _trainlist
            testlist += _testlist
        self.trainlist = trainlist
        self.testlist = testlist
        return trainlist, testlist

'''
    def autoTag(self, img, eps=0.005, epoch=None):
        self.phase = 'inference'
        self.net.eval()
        torch.set_grad_enabled(False)
        self.load_weights(epoch)
        hi, wi, _ = img.shape
        imgsegres = self.deal(img)
        
        img_ipt = self.img_preprocess(img)
        output = self.net(img_ipt) # multi: (1,nclass-1,h,w)
        output = output[0, ...].cpu().numpy()
        output = np.transpose(output, (1,2,0))
        output = np.rint(output).astype(int) # multi: (h,w,nclass-1)
        
        contour_list = {}
        for i in range(self.nclass-1):
            contour_list[self.class_table[i+1]] = []
            
            layer = output[..., i]
            image = 255*layer.copy().astype(np.uint8)
            image = cv2.resize(image, (wi, hi), cv2.INTER_NEAREST)
            
            _, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            for cont in contours:
                epsilon = eps*cv2.arcLength(cont,True)
                box = cv2.approxPolyDP(cont,epsilon,True)
                imgsegres = cv2.polylines(imgsegres,[box],True,(0,0,255),3)
                contour_list[self.class_table[i+1]].append(box)
        
        return imgsegres, contour_list
'''

class SegNeti(SegNet):
    def __init__(self, wpath, inputshape, device='cuda', mode='lanenet', num_class=2):
        self.num_class = num_class
        self.device = device
        self.rgb_means = (127.5,127.5,127.5)
        self.input_shape = inputshape
        self.colormap = [[0,0,0], [0,255,255], [0,0,255], [0,255,0], [255,0,0]]
        self.labelmode = 'single'

        if mode == 'huge':
            self.net = BuildNet('Seg16', num_class, 3, inputshape, 32, 'RFBResBlockPlus', 
                                1, 'conv', 'conv', softmax=False)
        elif mode == 'lanenet':
            model_init_nclass = num_class if self.labelmode=='single' else num_class-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = LANENET_NOVT(num_classes=model_init_nclass, sigmoid=do_sigmoid)
        elif mode == 'lanenet_big':
            model_init_nclass = num_class if self.labelmode=='single' else num_class-1
            do_sigmoid = True if self.labelmode=='multi' else False
            self.net = LANENET_BIG(num_classes=model_init_nclass, sigmoid=do_sigmoid)
        else:
            raise ValueError
        
        self.net.load_state_dict(torch.load(wpath, map_location='cuda:0'))
        # self.net.state_dict().update(torch.load(wpath))
        
        self.net = self.net.to(self.device)
        
        self.phase = 'inference'
        self.net.eval()
        torch.set_grad_enabled(False)
        



