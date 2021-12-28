import sys
from collections import defaultdict
from dtw_c import dtw_c as dtw
import torch
from torch.autograd import Variable
from torch import nn

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.train_tools import *
from Basic.utils.hdf5_tools import *
from Basic.utils.profiler_tools import *

from VC.utils.cycle_vae.loss_tools import *
from VC.utils.cycle_vae.train_tools import plot_tool_cycle_vae, save_checkpoint_cycle_vae

sys.path.insert(0, '/home/huanyuan/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
from common.common.utils.python.logging_helpers import setup_logger


def iter_batch(cfg, batch):
    # init
    n_cyc = cfg.net.yaml['cycle_vae_params']['n_cyc']
    
    batch_frame = {}
    batch_frame['flen_src'] = batch['flen_src'].data.numpy()
    max_flen = np.max(batch_frame['flen_src']) ## get max frame length
    batch_frame['flen_spc_src'] = batch['flen_spc_src'].data.numpy()
    max_flen_spc_src = np.max(batch_frame['flen_spc_src']) ## get max frame length

    batch_frame['flen_trg'] = batch['flen_trg'].data.numpy()
    max_flen_trg = np.max(batch_frame['flen_trg']) ## get max frame length
    batch_frame['flen_spc_trg'] = batch['flen_spc_trg'].data.numpy()
    max_flen_spc_trg = np.max(batch_frame['flen_spc_trg']) ## get max frame length

    # src
    batch_frame['h_src'] = batch['h_src'][:, :max_flen].cuda()
    batch_frame['code_src'] = batch['code_src'][:, :max_flen].cuda()
    batch_frame['class_code_src'] = batch['class_code_src'][: ,:max_flen].cuda()
    batch_frame['spcidx_src'] = batch['spcidx_src'][: ,:max_flen_spc_src].cuda()

    # trg
    batch_frame['h_trg'] = batch['h_trg'][:,:max_flen_trg].cuda()
    batch_frame['spcidx_trg'] = batch['spcidx_trg'][:,:max_flen_spc_trg].cuda()

    batch_frame['hdf5_name_src'] = batch['hdf5_name_src']
    batch_frame['hdf5_name_trg'] = batch['hdf5_name_trg']
    batch_frame['spk_trg_list'] = batch['spk_trg_list']
    batch_frame['file_src_trg_flag'] = batch['file_src_trg_flag']
    batch_frame['pair_flag'] = True if True in batch_frame['file_src_trg_flag'] else False

    batch_frame['code_trg_list'] = [None] * n_cyc
    batch_frame['class_code_trg_list'] = [None] * n_cyc
    batch_frame['h_src2trg_list'] = [None] * n_cyc
    
    for i in range(n_cyc):
        batch_frame['code_trg_list'][i] = batch['code_trg_list'][i][:, :max_flen].cuda()
        batch_frame['class_code_trg_list'][i] = batch['class_code_trg_list'][i][:, :max_flen].cuda()
        batch_frame['h_src2trg_list'][i] = batch['h_src2trg_list'][i][:, :max_flen].cuda()
    
    # batch_size batch_frame_size
    batch_size = batch_frame['h_src'].size(0)
    batch_frame_size = cfg.train.batch_frame_size

    # init
    batch_frame['src_s_idx'] = 0
    batch_frame['src_e_idx'] = batch_frame_size - 1
    batch_frame['spcidcs_src_s_idx'] = np.repeat(-1, batch_size)
    batch_frame['spcidcs_src_e_idx'] = np.repeat(-1, batch_size)
    batch_frame['flen_acc'] = np.repeat(batch_frame_size, batch_size)
    s_flag = np.repeat(False, batch_size)
    e_flag = np.repeat(True, batch_size)

    for j in range(batch_size):
        for i in range(batch_frame['spcidcs_src_e_idx'][j] + 1, batch_frame['flen_spc_src'][j]):
            if not s_flag[j] and batch_frame['spcidx_src'][j,i] >= batch_frame['src_s_idx']:
                if batch_frame['spcidx_src'][j,i] > batch_frame['src_e_idx']:
                    batch_frame['spcidcs_src_s_idx'][j] = -1
                    break
                batch_frame['spcidcs_src_s_idx'][j] = i
                s_flag[j] = True
                e_flag[j] = False
                if i == batch_frame['flen_spc_src'][j] - 1:
                    batch_frame['spcidcs_src_e_idx'][j] = i
                    s_flag[j] = False
                    e_flag[j] = True
                    break
            elif not e_flag[j] and (batch_frame['spcidx_src'][j,i] >= batch_frame['src_e_idx'] or i == batch_frame['flen_spc_src'][j] - 1):
                if batch_frame['spcidx_src'][j,i] > batch_frame['src_e_idx']:
                    batch_frame['spcidcs_src_e_idx'][j] = i-1
                else:
                    batch_frame['spcidcs_src_e_idx'][j] = i
                s_flag[j] = False
                e_flag[j] = True
                break
    
    # 产生第一帧
    batch_frame['select_utt_idx'] = [i for i in range(batch_size)]
    batch_frame['end_bool'] = False
    yield batch_frame

    while batch_frame['src_e_idx'] < max_flen - 1:
        batch_frame['src_s_idx'] = batch_frame['src_e_idx'] + 1
        batch_frame['src_e_idx'] = batch_frame['src_s_idx'] + batch_frame_size - 1
        if batch_frame['src_e_idx'] >= max_flen:
            batch_frame['src_e_idx'] = max_flen - 1

        batch_frame['select_utt_idx']  = []
        for j in range(batch_size):
            if batch_frame['spcidcs_src_e_idx'][j] < batch_frame['flen_spc_src'][j] - 1:
                if batch_frame['src_e_idx'] >= batch_frame['flen_src'][j]:
                    batch_frame['flen_acc'][j] = batch_frame['flen_src'][j] - batch_frame['src_s_idx']
                for i in range(batch_frame['spcidcs_src_e_idx'][j]+1,batch_frame['flen_spc_src'][j]):
                    if not s_flag[j] and batch_frame['spcidx_src'][j,i] >= batch_frame['src_s_idx']:
                        if batch_frame['spcidx_src'][j,i] > batch_frame['src_e_idx']:
                            batch_frame['spcidcs_src_s_idx'][j] = -1
                            break
                        batch_frame['spcidcs_src_s_idx'][j] = i
                        s_flag[j] = True
                        e_flag[j] = False
                        if i == batch_frame['flen_spc_src'][j]-1:
                            batch_frame['spcidcs_src_e_idx'][j] = i
                            s_flag[j] = False
                            e_flag[j] = True
                            break
                    elif not e_flag[j] and (batch_frame['spcidx_src'][j,i] >= batch_frame['src_e_idx'] or \
                                                i == batch_frame['flen_spc_src'][j]-1):
                        if batch_frame['spcidx_src'][j,i] > batch_frame['src_e_idx']:
                            batch_frame['spcidcs_src_e_idx'][j] = i-1
                        else:
                            batch_frame['spcidcs_src_e_idx'][j] = i
                        s_flag[j] = False
                        e_flag[j] = True
                        break
                batch_frame['select_utt_idx'].append(j)

        # 滑窗产生后续帧
        batch_frame['end_bool'] = False
        yield batch_frame

    # 滑窗结束，产生结束帧
    batch_frame['end_bool'] = True
    yield batch_frame

    
class CycleVae():
    
    def __init__(self, cfg):

        # init
        self.cfg = cfg
        self.batch_size = self.cfg.train.batch_size

        self.n_cyc = self.cfg.net.yaml['cycle_vae_params']['n_cyc']
        self.stdim = self.cfg.net.yaml['cycle_vae_params']['stdim']
        self.lat_dim = self.cfg.net.yaml['cycle_vae_params']['lat_dim']
        self.n_spk = self.cfg.net.yaml['cycle_vae_params']['spk_dim']
        self.arparam = self.cfg.net.yaml['encoder_params']['arparam']
        dataset_name = '_'.join([cfg.general.dataset_list[idx] for idx in range(len(cfg.general.dataset_list))])
        self.stats_jnt_path = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5', dataset_name, 'world', f"stats_jnt.h5")

        # logging
        self.logging_init()

        # model 
        self.model_init()

        # loss
        self.loss_init()

        # optimizer
        self.optimizer_init()

        # show
        self.show_info()

        # profiler
        self.profiler_init()
        
        # state
        self.state_init()
        
        # variable
        self.variable_init()


    def logging_init(self):

        # enable logging
        self.log_file = os.path.join(self.cfg.general.save_dir, 'logging', 'train_log.txt')
        self.logger = setup_logger(self.log_file, 'vc_train')

        return 

    
    def model_init(self):
        
        self.model = {}
        self.model['encoder'] = import_network(self.cfg, self.cfg.net.encoder_model_name, self.cfg.net.encoder_class_name)
        self.model['decoder'] = import_network(self.cfg, self.cfg.net.decoder_model_name, self.cfg.net.decoder_class_name)
        
        # joint normalize statistics 
        self.mean_jnt = torch.FloatTensor(read_hdf5(self.stats_jnt_path, "mean_feat_org_lf0"))
        self.std_jnt = torch.FloatTensor(read_hdf5(self.stats_jnt_path, "scale_feat_org_lf0"))
        self.mean_jnt_trg = torch.FloatTensor(read_hdf5(self.stats_jnt_path, "mean_feat_org_lf0")[self.stdim: ])
        self.std_jnt_trg = torch.FloatTensor(read_hdf5(self.stats_jnt_path, "scale_feat_org_lf0")[self.stdim: ])

        if torch.cuda.is_available():
            self.mean_jnt = self.mean_jnt.cuda()
            self.std_jnt = self.std_jnt.cuda()
            self.mean_jnt_trg = self.mean_jnt_trg.cuda()
            self.std_jnt_trg = self.std_jnt_trg.cuda()
        else:
            raise Exception("gpu is not available. please check the setting.")

        self.model['encoder'].module.scale_in.weight = torch.nn.Parameter(torch.diag(1.0/self.std_jnt.data).unsqueeze(2))               # 将均值和方差放在 1 * 1 卷积中计算（漂亮！网络训练速度更快，注意：param.requires_grad = False，不能进行梯度更新）
        self.model['encoder'].module.scale_in.bias = torch.nn.Parameter(-(self.mean_jnt.data/self.std_jnt.data))
        self.model['decoder'].module.scale_out.weight = torch.nn.Parameter(torch.diag(self.std_jnt_trg.data).unsqueeze(2))              # 反归一化，反均值和方差，思考这里使用反归一化的目的和意义（用于还原构建音频）
        self.model['decoder'].module.scale_out.bias = torch.nn.Parameter(self.mean_jnt_trg.data)
        
        return
    
    
    def loss_init(self):
        
        self.criterion = {}
        self.criterion["mcd"] = MCDloss()
        self.criterion["ce"] = nn.CrossEntropyLoss()

        return


    def optimizer_init(self):
        
        for param in self.model['encoder'].parameters():
            param.requires_grad = True
        for param in self.model['decoder'].parameters():
            param.requires_grad = True
        for param in self.model['encoder'].module.scale_in.parameters():
            param.requires_grad = False
        for param in self.model['decoder'].module.scale_out.parameters():
            param.requires_grad = False

        module_list = list(self.model['encoder'].module.conv.parameters())
        module_list += list(self.model['encoder'].module.gru.parameters()) + list(self.model['encoder'].module.out_1.parameters())
        module_list += list(self.model['decoder'].module.conv.parameters())
        module_list += list(self.model['decoder'].module.gru.parameters()) + list(self.model['decoder'].module.out_1.parameters())

        self.optimizer = torch.optim.Adam(module_list, lr=self.cfg.optimizer.lr)

        return 

    
    def show_info(self):

        parameters = filter(lambda p: p.requires_grad, self.model['encoder'].parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        self.logger.info('Trainable Parameters (encoder): %.3f million' % parameters)
        parameters = filter(lambda p: p.requires_grad, self.model['decoder'].parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        self.logger.info('Trainable Parameters (decoder): %.3f million' % parameters)

        
        return 


    def profiler_init(self):

        # profiler
        self.profiler = Profiler(summarize_every=self.cfg.train.show_profiler, disabled=False)

        # loss dict
        self.loss_dict = defaultdict(float)

        # epoch loss dict
        self.epoch_loss_dict = defaultdict(float)

        return


    def state_init(self):
        
        # train
        if self.arparam:
            self.init_pp = np.zeros((self.batch_size, 1, self.lat_dim * 2 + self.n_spk ))
        else:
            self.init_pp = np.zeros((self.batch_size, 1, self.lat_dim + self.n_spk))

        self.y_in_pp = torch.FloatTensor(self.init_pp).cuda()
        self.y_in_src = self.y_in_trg = \
            torch.unsqueeze(torch.unsqueeze((0 - self.mean_jnt_trg) / self.std_jnt_trg, 0), 0).repeat(self.batch_size, 1, 1)

        # eval
        with torch.no_grad():
            if self.arparam:
                init_pp_eval = np.zeros((1, 1, self.lat_dim * 2 + self.n_spk))
            else:
                init_pp_eval = np.zeros((1, 1, self.lat_dim + self.n_spk))

            self.y_in_pp_eval = torch.FloatTensor(init_pp_eval).cuda()
            self.y_in_src_eval = self.y_in_trg_eval = torch.unsqueeze(\
                torch.unsqueeze((0 - self.mean_jnt_trg) / self.std_jnt_trg, 0), 0).repeat(1, 1, 1)
        
        return


    def variable_init(self):

        self.variable = {}
        # src encoder
        self.variable['batch_lat_src'] = [None] * self.n_cyc
        self.variable['batch_mle_lat_src'] = [None] * self.n_cyc
        self.variable['batch_latpost_src'] = [None] * self.n_cyc
        self.variable['batch_scpost_src'] = [None] * self.n_cyc
        self.variable['y_in_pp_src'] = [None] * self.n_cyc
        self.variable['h_in_pp_src'] = [None] * self.n_cyc

        # src encoder src decoder
        self.variable['batch_trj_src_src'] = [None] * self.n_cyc
        self.variable['y_in_src_src'] = [None] * self.n_cyc
        self.variable['h_in_src_src'] = [None] * self.n_cyc

        # src encoder trg decoder
        self.variable['batch_trj_src_trg'] = [None] * self.n_cyc
        self.variable['y_in_src_trg'] = [None] * self.n_cyc
        self.variable['h_in_src_trg'] = [None] * self.n_cyc

        # trg encoder 
        self.variable['batch_lat_src_trg'] = [None] * self.n_cyc
        self.variable['batch_mle_lat_src_trg'] = [None] * self.n_cyc
        self.variable['batch_latpost_src_trg'] = [None] * self.n_cyc
        self.variable['batch_scpost_src_trg'] = [None] * self.n_cyc
        self.variable['y_in_pp_src_trg'] = [None] * self.n_cyc
        self.variable['h_in_pp_src_trg'] = [None] * self.n_cyc

        # trg encoder src decoder
        self.variable['batch_trj_src_trg_src'] = [None] * self.n_cyc
        self.variable['y_in_src_trg_src'] = [None] * self.n_cyc
        self.variable['h_in_src_trg_src'] = [None] * self.n_cyc

        # loss
        self.variable['batch_loss_mcd_trg_trg'] = [None] * self.n_cyc
        self.variable['batch_loss_mcd_trg_src_trg'] = [None] * self.n_cyc
        self.variable['batch_loss_mcd_trg_src'] = [None] * self.n_cyc
        
        self.variable['batch_loss_mcd_src_src'] = [None] * self.n_cyc
        self.variable['batch_loss_mcd_src_trg_src'] = [None] * self.n_cyc
        self.variable['batch_loss_mcd_src_trg'] = [None] * self.n_cyc

        self.variable['batch_loss_lat_src'] = [None] * self.n_cyc
        self.variable['batch_loss_lat_trg'] = [None] * self.n_cyc

        self.variable['batch_loss_scpost_src'] = [None] * self.n_cyc
        self.variable['batch_loss_scpost_trg'] = [None] * self.n_cyc

        self.variable['batch_loss_lat_src_cv'] = [None] * self.n_cyc
        self.variable['batch_loss_lat_trg_cv'] = [None] * self.n_cyc

        self.variable['batch_loss_scpost_src_cv'] = [None] * self.n_cyc
        self.variable['batch_loss_scpost_trg_cv'] = [None] * self.n_cyc

        return 


    def modle_forward_new_barch(self):

        # new batch sequence, previous output and hidden state are zeros initialized
        for i in range(self.n_cyc):

            # encoding input features
            if i > 0: # [2nd, 3rd, ..., Nth] cycle
                self.variable['batch_lat_src'][i], \
                batch_param, \
                self.variable['y_in_pp_src'][i], \
                self.variable['h_in_pp_src'][i], \
                self.variable['batch_mle_lat_src'][i] = \
                    self.model['encoder'](torch.cat((self.batch_frame['h_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1, :self.stdim], \
                        self.variable['batch_trj_src_trg_src'][i-1]), 2), \
                        self.y_in_pp, do=True)
            else: # 1st cycle
                self.variable['batch_lat_src'][i], \
                batch_param, \
                self.variable['y_in_pp_src'][i], \
                self.variable['h_in_pp_src'][i], \
                self.variable['batch_mle_lat_src'][i] = \
                    self.model['encoder'](self.batch_frame['h_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                        self.y_in_pp, do=True)
            self.variable['batch_scpost_src'][i] = batch_param[:, :, :self.n_spk] # speaker_posterior    说话人结果 torch.Size([5, 80, 14])
            self.variable['batch_latpost_src'][i] = batch_param[:, :, self.n_spk:] # latent_posterior    分布均值方差结果 torch.Size([5, 80, 64])

            # spectral reconstruction
            self.variable['batch_trj_src_src'][i], \
            self.variable['y_in_src_src'][i], \
            self.variable['h_in_src_src'][i] = \
                self.model['decoder'](torch.cat((self.batch_frame['code_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_lat_src'][i]), 2), \
                    self.y_in_src, do=True)

            # spectral conversion
            self.variable['batch_trj_src_trg'][i], \
            self.variable['y_in_src_trg'][i], \
            self.variable['h_in_src_trg'][i] = \
                self.model['decoder'](torch.cat((self.batch_frame['code_trg_list'][i][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_lat_src'][i]), 2), \
                    self.y_in_trg, do=True)

            # encoding converted features
            self.variable['batch_lat_src_trg'][i], \
            batch_param, \
            self.variable['y_in_pp_src_trg'][i], \
            self.variable['h_in_pp_src_trg'][i], \
            self.variable['batch_mle_lat_src_trg'][i] = \
                self.model['encoder'](torch.cat((self.batch_frame['h_src2trg_list'][i][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_trj_src_trg'][i]) ,2), \
                    self.y_in_pp, do=True)
            self.variable['batch_scpost_src_trg'][i] = batch_param[:, :, :self.n_spk] #speaker_posterior
            self.variable['batch_latpost_src_trg'][i] = batch_param[:, :, self.n_spk:] #latent_posterior

            # cyclic spectral reconstruction
            self.variable['batch_trj_src_trg_src'][i], \
            self.variable['y_in_src_trg_src'][i], \
            self.variable['h_in_src_trg_src'][i] = \
                self.model['decoder'](torch.cat((self.batch_frame['code_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_lat_src_trg'][i]), 2), \
                    self.y_in_src, do=True)               # 源说话人的标签

            # compute utterance-level accuracy if this is not the 1st batch in the epoch
            # 2064 - 2089
            # if iter_count > 0:
            #     for j in range(n_batch_utt):
            #         spk_src_ = os.path.basename(os.path.dirname(prev_featfile_src[j]))
            #         for k in range(n_spk):
            #             if spk_src_ == spk_list[k]:
            #                 #GV stat of reconst.
            #                 gv_src_src[0][k].append(torch.var(\
            #                     tmp_src_src[j,:prev_flens_src[j]], 0).cpu().data.numpy())
            #                 #GV stat of cyclic reconst.
            #                 gv_src_trg_src[0][k].append(torch.var(\
            #                     tmp_src_trg_src[j,:prev_flens_src[j]], 0).cpu().data.numpy())
            #                 break
            #         spk_trg_ = prev_pair_spk[i][j] #find the target pair in prev. batch
            #         for k in range(n_spk):
            #             if spk_trg_ == spk_list[k]:
            #                 #GV stat of converted
            #                 gv_src_trg[0][k].append(torch.var(\
            #                     tmp_src_trg[j,:prev_flens_src[j]], 0).cpu().data.numpy())
            #                 break

            # # store est. data from the 1st cycle for accuracy on utterance-level
            # tmp_src_src = self.variable['batch_trj_src_src'][0][:,:,1:]
            # tmp_src_trg = self.variable['batch_trj_src_trg'][0][:,:,1:]
            # tmp_src_trg_src = self.variable['batch_trj_src_trg_src'][0][:,:,1:]
            # trj_src_trg = self.variable['batch_trj_src_trg'][0]
            # trj_lat_src = self.variable['batch_mle_lat_src'][0]

        return 


    def modle_forward_continue(self):

        for i in range(self.n_cyc):
            # encoding input features
            if i > 0: # [2nd, 3rd, ..., Nth] cycle
                self.variable['batch_lat_src'][i],\
                batch_param, \
                self.variable['y_in_pp_src'][i], \
                self.variable['h_in_pp_src'][i], \
                self.variable['batch_mle_lat_src'][i] = \
                    self.model['encoder'](torch.cat((self.batch_frame['h_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1, :self.stdim], \
                        self.variable['batch_trj_src_trg_src'][i-1]), 2), \
                        Variable(self.variable['y_in_pp_src'][i].data).detach(), \
                        h_in=Variable(self.variable['h_in_pp_src'][i].data).detach(), do=True)
            else: # 1st cycle
                self.variable['batch_lat_src'][i], \
                batch_param, self.variable['y_in_pp_src'][i], \
                self.variable['h_in_pp_src'][i], \
                self.variable['batch_mle_lat_src'][i] = \
                    self.model['encoder'](self.batch_frame['h_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                        Variable(self.variable['y_in_pp_src'][0].data).detach(), \
                        h_in=Variable(self.variable['h_in_pp_src'][0].data).detach(), do=True)

            self.variable['batch_scpost_src'][i] = batch_param[:, :, :self.n_spk] #speaker_posterior
            self.variable['batch_latpost_src'][i] = batch_param[:, :, self.n_spk:] #latent_posterior

            # spectral reconstruction
            self.variable['batch_trj_src_src'][i], \
            self.variable['y_in_src_src'][i], \
            self.variable['h_in_src_src'][i] = \
                self.model['decoder'](torch.cat((self.batch_frame['code_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_lat_src'][i]),2), \
                    Variable(self.variable['y_in_src_src'][i].data).detach(), \
                    h_in=Variable(self.variable['h_in_src_src'][i].data).detach(), do=True)

            # spectral conversion
            self.variable['batch_trj_src_trg'][i], \
            self.variable['y_in_src_trg'][i], \
            self.variable['h_in_src_trg'][i] = \
                self.model['decoder'](torch.cat((self.batch_frame['code_trg_list'][i][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_lat_src'][i]),2), \
                    Variable(self.variable['y_in_src_trg'][i].data).detach(), \
                    h_in=Variable(self.variable['h_in_src_trg'][i].data).detach(), do=True)

            # encoding converted features
            self.variable['batch_lat_src_trg'][i], \
            batch_param, \
            self.variable['y_in_pp_src_trg'][i], \
            self.variable['h_in_pp_src_trg'][i], \
            self.variable['batch_mle_lat_src_trg'][i] = \
                self.model['encoder'](torch.cat((self.batch_frame['h_src2trg_list'][i][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_trj_src_trg'][i]),2), \
                    Variable(self.variable['y_in_pp_src_trg'][i].data).detach(), \
                    h_in=Variable(self.variable['h_in_pp_src_trg'][i].data).detach(), do=True)
            self.variable['batch_scpost_src_trg'][i] = batch_param[:,:,:self.n_spk] #speaker_posterior
            self.variable['batch_latpost_src_trg'][i] = batch_param[:,:,self.n_spk:] #latent_posterior

            # cyclic spectral reconstruction
            self.variable['batch_trj_src_trg_src'][i], \
            self.variable['y_in_src_trg_src'][i], \
            self.variable['h_in_src_trg_src'][i] = \
                self.model['decoder'](torch.cat((self.batch_frame['code_src'][:, self.batch_frame['src_s_idx']:self.batch_frame['src_e_idx'] + 1], \
                    self.variable['batch_lat_src_trg'][i]),2), \
                    Variable(self.variable['y_in_src_trg_src'][i].data).detach(), \
                    h_in=Variable(self.variable['h_in_src_trg_src'][i].data).detach(), do=True)

            # # store est. data from the 1st cycle for accuracy on utterance-level
            # tmp_src_src = torch.cat((tmp_src_src, self.variable['batch_trj_src_src'][0][:,:,1:]), 1)
            # tmp_src_trg = torch.cat((tmp_src_trg, self.variable['batch_trj_src_trg'][0][:,:,1:]), 1)
            # tmp_src_trg_src = torch.cat((tmp_src_trg_src, self.variable['batch_trj_src_trg_src'][0][:,:,1:]), 1)
            # trj_src_trg = torch.cat((trj_src_trg, self.variable['batch_trj_src_trg'][0]), 1)
            # trj_lat_src = torch.cat((trj_lat_src, self.variable['batch_mle_lat_src'][0]), 1)

        return


    def calculate_loss(self):

        for i in range(self.n_cyc): # iterate over all cycles
            
            for k, j in enumerate(self.batch_frame['select_utt_idx']): # iterate over all valid utterances
                
                src_e_idx_valid = self.batch_frame['src_s_idx'] + self.batch_frame['flen_acc'][j] # valid length/idcs of current utt. for optim.

                # valid spectral segment for optim.
                batch_src_optim = self.batch_frame['h_src'][j, self.batch_frame['src_s_idx']: src_e_idx_valid, self.stdim:]

                # mel-cepstral distortion (MCD)-based L1-loss of spectral features
                # spectral reconstruction
                _, tmp_batch_loss_mcd_src_src, _ = self.criterion["mcd"](self.variable['batch_trj_src_src'][i][j, :self.batch_frame['flen_acc'][j]], batch_src_optim)
                # cyclic spectral reconstruction
                _, tmp_batch_loss_mcd_src_trg_src, _ = self.criterion["mcd"](self.variable['batch_trj_src_trg_src'][i][j, :self.batch_frame['flen_acc'][j]], batch_src_optim)
                # spectral conversion
                _, tmp_batch_loss_mcd_src_trg, _ = self.criterion["mcd"](self.variable['batch_trj_src_trg'][i][j, :self.batch_frame['flen_acc'][j]], batch_src_optim)

                # cross-entropy (CE) of speaker-posterior
                # encoding
                tmp_batch_loss_scpost_src = self.criterion["ce"](self.variable['batch_scpost_src'][i][j, :self.batch_frame['flen_acc'][j]], \
                    self.batch_frame['class_code_src'][j, self.batch_frame['src_s_idx']: src_e_idx_valid])
                # encoding converted
                tmp_batch_loss_scpost_src_cv = self.criterion["ce"](self.variable['batch_scpost_src_trg'][i][j, :self.batch_frame['flen_acc'][j]], \
                    self.batch_frame['class_code_trg_list'][i][j, self.batch_frame['src_s_idx'] : src_e_idx_valid])

                # KL-divergence of latent-posterior to the standard Laplacian prior 标准拉普拉斯先验
                # encoding
                tmp_batch_loss_lat_src = loss_vae_laplace(\
                    self.variable['batch_latpost_src'][i][j,:self.batch_frame['flen_acc'][j]], lat_dim=self.lat_dim, clip=True)
                # encoding converted
                tmp_batch_loss_lat_src_cv = loss_vae_laplace(\
                    self.variable['batch_latpost_src_trg'][i][j,:self.batch_frame['flen_acc'][j]], lat_dim=self.lat_dim, clip=True)

                if k > 0:
                    self.variable['batch_loss_mcd_src_src'][i] = torch.cat((self.variable['batch_loss_mcd_src_src'][i], \
                        tmp_batch_loss_mcd_src_src.unsqueeze(0)))
                    self.variable['batch_loss_mcd_src_trg_src'][i] = torch.cat((self.variable['batch_loss_mcd_src_trg_src'][i], \
                        tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)))
                    self.variable['batch_loss_mcd_src_trg'][i] = torch.cat((self.variable['batch_loss_mcd_src_trg'][i], \
                        tmp_batch_loss_mcd_src_trg.unsqueeze(0)))
                    self.variable['batch_loss_scpost_src'][i] = torch.cat((self.variable['batch_loss_scpost_src'][i], \
                        tmp_batch_loss_scpost_src.unsqueeze(0)))
                    # use 1st and 2nd scpost for 2nd scpost loss
                    self.variable['batch_loss_scpost_src_cv'][i] = torch.cat((self.variable['batch_loss_scpost_src_cv'][i], \
                        tmp_batch_loss_scpost_src.unsqueeze(0), \
                        tmp_batch_loss_scpost_src_cv.unsqueeze(0)))
                    self.variable['batch_loss_lat_src'][i] = torch.cat((self.variable['batch_loss_lat_src'][i], \
                        tmp_batch_loss_lat_src.unsqueeze(0)))
                    # use 1st and 2nd latent for 2nd latent loss
                    self.variable['batch_loss_lat_src_cv'][i] = torch.cat((self.variable['batch_loss_lat_src_cv'][i], \
                                                    tmp_batch_loss_lat_src.unsqueeze(0), \
                                                    tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                else:
                    self.variable['batch_loss_mcd_src_src'][i] = tmp_batch_loss_mcd_src_src.unsqueeze(0)
                    self.variable['batch_loss_mcd_src_trg_src'][i] = tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)
                    self.variable['batch_loss_mcd_src_trg'][i] = tmp_batch_loss_mcd_src_trg.unsqueeze(0)
                    self.variable['batch_loss_scpost_src'][i] = tmp_batch_loss_scpost_src.unsqueeze(0)
                    # use 1st and 2nd scpost for 2nd scpost loss
                    self.variable['batch_loss_scpost_src_cv'][i] = torch.cat((tmp_batch_loss_scpost_src.unsqueeze(0), \
                                                        tmp_batch_loss_scpost_src_cv.unsqueeze(0)))
                    self.variable['batch_loss_lat_src'][i] = tmp_batch_loss_lat_src.unsqueeze(0)
                    # use 1st and 2nd latent for 2nd latent loss
                    self.variable['batch_loss_lat_src_cv'][i] = torch.cat((tmp_batch_loss_lat_src.unsqueeze(0), \
                                                    tmp_batch_loss_lat_src_cv.unsqueeze(0)))

                # record the loss statistics
                self.epoch_loss_dict["epoch_{}/idx[{}]".format(self.mode, i)] += 1
                self.epoch_loss_dict["epoch_{}/loss_mcd_src_src[{}]".format(self.mode, i)] += tmp_batch_loss_mcd_src_src.item()
                self.epoch_loss_dict["epoch_{}/loss_mcd_src_trg_src[{}]".format(self.mode, i)] += tmp_batch_loss_mcd_src_trg_src.item()
                self.epoch_loss_dict["epoch_{}/loss_mcd_src_trg[{}]".format(self.mode, i)] += tmp_batch_loss_mcd_src_trg.item()
                self.epoch_loss_dict["epoch_{}/loss_scpost_src[{}]".format(self.mode, i)] += tmp_batch_loss_scpost_src.item()
                self.epoch_loss_dict["epoch_{}/loss_scpost_src_cv[{}]".format(self.mode, i)] += tmp_batch_loss_scpost_src_cv.item()
                self.epoch_loss_dict["epoch_{}/loss_lat_src_cv[{}]".format(self.mode, i)] += tmp_batch_loss_lat_src_cv.item()
                self.epoch_loss_dict["epoch_{}/loss_lat_src[{}]".format(self.mode, i)] += tmp_batch_loss_lat_src.item()
                
            # accumulate loss for optimization
            if i > 0: # [2nd, 3rd, ..., Nth] cycle
                self.batch_loss += \
                    self.variable['batch_loss_mcd_src_src'][i].sum() + self.variable['batch_loss_mcd_src_trg_src'][i].sum() + \
                    self.variable['batch_loss_lat_src'][i].sum() + self.variable['batch_loss_lat_src_cv'][i].sum() + \
                    self.variable['batch_loss_scpost_src'][i].sum() + self.variable['batch_loss_scpost_src_cv'][i].sum()
            else: # 1st cycle
                if self.n_cyc > 0:
                    self.batch_loss = \
                        self.variable['batch_loss_mcd_src_src'][0].sum() + self.variable['batch_loss_mcd_src_trg_src'][0].sum() + \
                        self.variable['batch_loss_lat_src'][0].sum() + self.variable['batch_loss_lat_src_cv'][0].sum() + \
                        self.variable['batch_loss_scpost_src'][0].sum() + self.variable['batch_loss_scpost_src_cv'][0].sum()
                else: # only 1st half of the cycle due to n_cyc = 0 (VAE)
                    self.batch_loss = \
                        self.variable['batch_loss_mcd_src_src'][0].sum() + \
                        self.variable['batch_loss_lat_src'][0].sum() + \
                        self.variable['batch_loss_scpost_src'][0].sum()
            
            # record mean-loss from all utterances in this batch
            self.loss_dict["{}/batch_loss_mcd_src_src[{}]".format(self.mode, i)] += torch.mean(self.variable['batch_loss_mcd_src_src'][i]).item()
            self.loss_dict["{}/batch_loss_mcd_src_trg_src[{}]".format(self.mode, i)] += torch.mean(self.variable['batch_loss_mcd_src_trg_src'][i]).item()
            self.loss_dict["{}/batch_loss_mcd_src_trg[{}]".format(self.mode, i)] += torch.mean(self.variable['batch_loss_mcd_src_trg'][i]).item()
            self.loss_dict["{}/batch_loss_scpost_src[{}]".format(self.mode, i)] += torch.mean(self.variable['batch_loss_scpost_src'][i]).item()
            self.loss_dict["{}/batch_loss_scpost_src_cv[{}]".format(self.mode, i)] += torch.mean(self.variable['batch_loss_scpost_src_cv'][i]).item()
            self.loss_dict["{}/batch_loss_lat_src[{}]".format(self.mode, i)] += torch.mean(self.variable['batch_loss_lat_src'][i]).item()
            self.loss_dict["{}/batch_loss_lat_src_cv[{}]".format(self.mode, i)] += torch.mean(self.variable['batch_loss_lat_src_cv'][i]).item()

        return 


    def show_data_info(self):

        for i in range(self.batch_size):
            self.logger.info("%s %d %d %d %d %d %d %d %d" % (\
                self.batch_frame['hdf5_name_src'][i], \
                self.batch_frame['flen_src'][i], 
                self.batch_frame['flen_spc_src'][i], \
                self.batch_frame['src_s_idx'], \
                self.batch_frame['src_e_idx'], \
                self.batch_frame['spcidcs_src_s_idx'][i], \
                self.batch_frame['spcidcs_src_e_idx'][i], \
                self.batch_frame['spcidx_src'][i, self.batch_frame['spcidcs_src_s_idx'][i]].item(), \
                self.batch_frame['spcidx_src'][i, self.batch_frame['spcidcs_src_e_idx'][i]].item()))

        return 


    def update_loss_info(self):

        # compute MCD-based L2-loss (true MCD values) of (cyclic) reconst. spectra
        for j in self.batch_frame['select_utt_idx']: # iterate over all valid utterances in optim.

            if self.batch_frame['spcidcs_src_s_idx'][j] >= 0: # calculate MCD only with speech frames
                
                for i in range(self.n_cyc): # iterate over all cycles

                    # reconst. MCD with 0th power
                    tmp_batch_mcdpow_src_src, _ = dtw.calc_mcd(\
                        np.array(torch.index_select(self.batch_frame['h_src'][j], 0, self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:self.batch_frame['spcidcs_src_e_idx'][j]+1])[:,\
                            self.stdim:].cpu().data.numpy(), dtype=np.float64), \
                        np.array(torch.index_select(self.variable['batch_trj_src_src'][i][j], 0, self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:\
                            self.batch_frame['spcidcs_src_e_idx'][j]+1] - self.batch_frame['src_s_idx']).cpu().data.numpy(), dtype=np.float64))
                            
                    # reconst. MCD w/o 0th power, i.e., [:,1:]
                    tmp_batch_mcd_src_src, _ = dtw.calc_mcd(\
                        np.array(torch.index_select(self.batch_frame['h_src'][j], 0, self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:self.batch_frame['spcidcs_src_e_idx'][j]+1])[:,\
                            self.stdim + 1:].cpu().data.numpy(), dtype=np.float64), \
                        np.array(torch.index_select(\
                            self.variable['batch_trj_src_src'][i][j],0,self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:self.batch_frame['spcidcs_src_e_idx'][j]+1]-\
                            self.batch_frame['src_s_idx'])[:,1:].cpu().data.numpy(), dtype=np.float64))

                    # cyclic reconst. MCD with 0th power
                    tmp_batch_mcdpow_src_trg_src, _ = dtw.calc_mcd(\
                        np.array(torch.index_select(self.batch_frame['h_src'][j],0,self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:self.batch_frame['spcidcs_src_e_idx'][j]+1])[:,\
                            self.stdim:].cpu().data.numpy(), dtype=np.float64), \
                        np.array(torch.index_select(\
                            self.variable['batch_trj_src_trg_src'][i][j],0,self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:self.batch_frame['spcidcs_src_e_idx'][j]+1]-\
                            self.batch_frame['src_s_idx']).cpu().data.numpy(), dtype=np.float64))

                    # cyclic reconst. MCD w/o 0th power, i.e., [:,1:]
                    tmp_batch_mcd_src_trg_src, _ = dtw.calc_mcd(\
                        np.array(torch.index_select(self.batch_frame['h_src'][j],0,self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:self.batch_frame['spcidcs_src_e_idx'][j]+1])[:,\
                            self.stdim + 1:].cpu().data.numpy(),\
                            dtype=np.float64), \
                        np.array(torch.index_select(\
                            self.variable['batch_trj_src_trg_src'][i][j],0,self.batch_frame['spcidx_src'][j,\
                            self.batch_frame['spcidcs_src_s_idx'][j]:self.batch_frame['spcidcs_src_e_idx'][j]+1]-\
                            self.batch_frame['src_s_idx'])[:,1:].cpu().data.numpy(), dtype=np.float64))
                    
                    # record loss statistics
                    self.loss_dict["{}/batch_mcdpow_src_src[{}]".format(self.mode, i)] += tmp_batch_mcdpow_src_src
                    self.loss_dict["{}/batch_mcd_src_src[{}]".format(self.mode, i)] += tmp_batch_mcd_src_src
                    self.loss_dict["{}/batch_mcdpow_src_trg_src[{}]".format(self.mode, i)] += tmp_batch_mcdpow_src_trg_src
                    self.loss_dict["{}/batch_mcd_src_trg_src[{}]".format(self.mode, i)] += tmp_batch_mcd_src_trg_src

                    self.epoch_loss_dict["epoch_{}/mcdpow_src_src[{}]".format(self.mode, i)] += tmp_batch_mcdpow_src_src
                    self.epoch_loss_dict["epoch_{}/mcd_src_src[{}]".format(self.mode, i)] += tmp_batch_mcd_src_src
                    self.epoch_loss_dict["epoch_{}/mcdpow_src_trg_src[{}]".format(self.mode, i)] += tmp_batch_mcdpow_src_trg_src
                    self.epoch_loss_dict["epoch_{}/mcd_src_trg_src[{}]".format(self.mode, i)] += tmp_batch_mcd_src_trg_src

        return 


    def train(self, train_dataloader, len_train_dataset):

        # init
        start_epoch, start_batch = 0, 0
        last_save_epoch, last_show_spoch, last_test_epoch = 0, 0, 0

        batch_number = len(train_dataloader)
        data_iter = iter(train_dataloader)
        batch_idx = start_batch
        iter_idx = 0
        
        msg = 'Training dataset number: {}'.format(len_train_dataset)
        self.logger.info(msg)

        # loop over batches
        for i in range(batch_number):
            self.mode = "train"

            epoch_idx = start_epoch + i * self.cfg.train.batch_size // len_train_dataset
            batch_idx += 1

            # Blocking, waiting for batch (threaded)
            batch = data_iter.next()

            self.model['encoder'].train()
            self.model['decoder'].train()

            generator_src = iter_batch(self.cfg, batch)
            
            while True:
                # Blocking, waiting for batch (threaded)
                self.batch_frame = next(generator_src)
                self.profiler.tick("Blocking, waiting for batch (threaded)")

                # Ending of batch_frame
                if self.batch_frame['end_bool']:
                    self.logger.info("Ending of batch_frame, going into next batch.")
                    self.profiler.tick("Forward pass")
                    self.profiler.tick("Generator Calculate Loss")
                    self.profiler.tick("Generator Backward pass")
                    self.profiler.tick("Generator Parameter update")
                    self.profiler.tick("Show information")
                    self.profiler.tick("Plot snapshot")

                    # Save model
                    if epoch_idx % self.cfg.train.save_epochs == 0 or epoch_idx == self.cfg.train.num_epochs - 1:
                        if last_save_epoch != epoch_idx:
                            last_save_epoch = epoch_idx

                            # save training model
                            save_checkpoint_cycle_vae(self.cfg, self.model, self.optimizer, epoch_idx, batch_idx)

                    # Show information
                    if last_show_spoch != epoch_idx:
                        last_show_spoch = epoch_idx

                        msg = 'epoch: {}, batch: {}, average optimization loss: '.format(epoch_idx, batch_idx)
                        for key in self.epoch_loss_dict.keys():
                            self.epoch_loss_dict[key] /= self.epoch_loss_dict["epoch_{}/idx".format(self.mode)]
                            msg += ', {}:{:.4f}'.format(str(key), self.epoch_loss_dict[key])
                        self.logger.info(msg)
                        self.epoch_loss_dict = defaultdict(float) # reset
                        self.profiler.tick("Show information")

                    break

                # Forward pass
                if self.batch_frame['src_s_idx'] > 0:
                    self.modle_forward_continue()
                else:
                    self.modle_forward_new_barch()
                self.profiler.tick("Forward pass")

                # 2090 - 2091
                # prev_featfile_src = featfile_src #record list of uttterances in current batch seq.
                # prev_pair_spk = pair_spk_list #record list of target speakers in current batch seq.

                ## optimization performed only for utterances with valid length, 
                ##   i.e., current mini-batch frames are still within the corresponding utterances 
                # check whether current batch has at least 1 utterance with valid length
                # if not, don't optimize at all (this is caused by having only ending silent frames)
                # 仅对有效长度的话语进行优化，select_utt_idx 用于调整 batch 中不同长短音频带来的影响（太短的音频训练批次较少）
                if not len(self.batch_frame['select_utt_idx']) > 0:
                    self.logger.info("There is no valid utterances in batch_frame, waiting for next batch.")
                    self.profiler.tick("Generator Calculate Loss")
                    self.profiler.tick("Generator Backward pass")
                    self.profiler.tick("Generator Parameter update")
                    self.profiler.tick("Show information")
                    self.profiler.tick("Plot snapshot")
                    continue
                
                # idx
                iter_idx += 1

                # Calculate loss
                self.calculate_loss()
                self.profiler.tick("Generator Calculate Loss")
    
                # Backward pass
                self.optimizer.zero_grad()
                self.batch_loss.backward()
                self.loss_dict["{}/batch_loss".format(self.mode)] += self.batch_loss.item()
                self.epoch_loss_dict["epoch_{}/batch_loss".format(self.mode)] += self.batch_loss.item()
                self.epoch_loss_dict["epoch_{}/idx".format(self.mode)] += 1
                self.profiler.tick("Generator Backward pass")

                # Parameter update
                self.optimizer.step()
                self.profiler.tick("Generator Parameter update")

                # Show information
                self.show_data_info()
                self.update_loss_info()

                msg = 'epoch: {}, batch: {}, iter: {}'.format(epoch_idx, batch_idx, iter_idx)
                for key in self.loss_dict.keys():
                    msg += ', {}:{:.4f}'.format(str(key), self.loss_dict[key])
                self.logger.info(msg)
                self.loss_dict = defaultdict(float) # reset
                self.profiler.tick("Show information")

                # Plot snapshot
                if (iter_idx % self.cfg.train.plot_snapshot) == 0:
                    plot_tool_cycle_vae(self.cfg, self.log_file)
                self.profiler.tick("Plot snapshot")
