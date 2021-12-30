import sys
import soundfile as sf
import pysptk as ps
import pyworld as pw

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset.audio import world_feature
from Basic.utils.hdf5_tools import *
from Basic.utils.train_tools import *

from VC.dataset.cyclevae.dataset_preload_audio_hdf5 import load_data_pd


class VcCycleVaeInfer():

    def __init__(self, cfg):

        # init
        self.cfg = cfg
        self.stdim = self.cfg.net.yaml['cyclevae_params']['stdim']
        self.lat_dim = self.cfg.net.yaml['cyclevae_params']['lat_dim']
        self.n_spk = self.cfg.net.yaml['cyclevae_params']['spk_dim']
        self.arparam = self.cfg.net.yaml['encoder_params']['arparam']
        dataset_name = '_'.join([cfg.general.dataset_list[idx] for idx in range(len(cfg.general.dataset_list))])
        self.stats_jnt_path = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5', dataset_name, 'world', f"stats_jnt.h5")
        self.hdf5_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_hdf5')
        self.state_hdf5_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5')

        # model
        self.model_init()

        # state
        self.state_init()

        # spk
        self.spk_init()


    def model_init(self):
        # load model
        self.model = {}
        self.model['encoder'] = import_network(self.cfg, self.cfg.net.encoder_model_name, self.cfg.net.encoder_class_name)
        self.model['decoder'] = import_network(self.cfg, self.cfg.net.decoder_model_name, self.cfg.net.decoder_class_name)

        # load state
        state_dict = load_state_dict(self.cfg.general.save_dir, self.cfg.test.model_epoch, 'checkpoints')
        self.model['encoder'].load_state_dict(state_dict["model"]["encoder"])
        self.model['decoder'].load_state_dict(state_dict["model"]["decoder"])

        # # load state test
        # state = torch.load("/home/huanyuan/code/third_code/vc/vcc20_baseline_cyclevae/baseline/egs/cyclevae/exp/tr50_cyclevae-mult-jnt-mix-scpost_laplace_vcc2020_24kHz_hl1_hu1024_ld32_kse3_dse2_ksd3_dsd2_cyc2_lr1e-4_bs80_do0.5_epoch80_bsu5_bsue35_nwrk2_pad2300_mine/checkpoint-78.pkl")

        # new_pre = {}
        # for k, v in state["model_encoder"].items():
        #     name = 'module.' + k
        #     new_pre[name] = v
        # self.model['encoder'].load_state_dict(new_pre, strict=False)

        # new_pre = {}
        # for k, v in state["model_decoder"].items():
        #     name = 'module.' + k
        #     new_pre[name] = v
        # self.model['decoder'].load_state_dict(new_pre, strict=False)
        
        self.model['encoder'].eval().cuda()
        self.model['decoder'].eval().cuda()

        # joint normalize statistics 
        self.mean_jnt_trg = torch.FloatTensor(read_hdf5(self.stats_jnt_path, "mean_feat_org_lf0")[self.stdim: ])
        self.std_jnt_trg = torch.FloatTensor(read_hdf5(self.stats_jnt_path, "scale_feat_org_lf0")[self.stdim: ])
        
        return


    def state_init(self):

        # infer
        with torch.no_grad():
            if self.arparam:
                init_pp = np.zeros((1, 1, self.lat_dim * 2 + self.n_spk))
            else:
                init_pp = np.zeros((1, 1, self.lat_dim + self.n_spk))

            self.y_in_pp = torch.FloatTensor(init_pp).cuda()
            self.y_in_src = self.y_in_trg = torch.unsqueeze(\
                torch.unsqueeze((0 - self.mean_jnt_trg) / self.std_jnt_trg, 0), 0).repeat(1, 1, 1)
        
        return

    
    def spk_init(self):
        
        self.data_pd = load_data_pd(self.cfg, hparams.TRAINING_NAME)
        self.spk_list = list(set(self.data_pd['speaker'].to_list()))
        self.spk_list.sort()
        assert self.n_spk == len(self.spk_list)
        self.spk_idx_dict = {}
        for i in range(self.n_spk):
            self.spk_idx_dict[self.spk_list[i]] = i
        
        return 


    def load_spk_info(self, spk_name_src, spk_name_trg):
        
        assert spk_name_src in self.spk_list
        assert spk_name_trg in self.spk_list

        spk_pd_src = self.data_pd[self.data_pd['speaker'] == spk_name_src]
        dataset_name_src = spk_pd_src['dataset'].to_list()[0]
        hdf5_normalize_path_src = os.path.join(self.state_hdf5_dir, dataset_name_src, 'world', f"stats_spk_{spk_name_src}.h5")
        self.f0_range_mean_src = read_hdf5(hdf5_normalize_path_src, "lf0_range_mean")
        self.f0_range_std_src = read_hdf5(hdf5_normalize_path_src, "lf0_range_std")
        self.code_idx_src = self.spk_idx_dict[spk_name_src]

        # load conf
        conf_path = os.path.join(self.hdf5_dir, dataset_name_src, 'conf', spk_name_src + '.f0')
        with open(conf_path, "r") as f :
            line = f.readline().strip()
            self.minf0 = float(line.split(' ')[0])
            self.maxf0 = float(line.split(' ')[1])

        conf_path = os.path.join(self.hdf5_dir, dataset_name_src, 'conf', spk_name_src + '.pow')
        with open(conf_path, "r") as f :
            line = f.readline().strip()
            self.pow = float(line.split(' ')[0])

        spk_pd_trg = self.data_pd[self.data_pd['speaker'] == spk_name_trg]
        dataset_name_trg = spk_pd_trg['dataset'].to_list()[0]
        hdf5_normalize_path_trg = os.path.join(self.state_hdf5_dir, dataset_name_trg, 'world', f"stats_spk_{spk_name_trg}.h5")
        self.f0_range_mean_trg = read_hdf5(hdf5_normalize_path_trg, "lf0_range_mean")
        self.f0_range_std_trg = read_hdf5(hdf5_normalize_path_trg, "lf0_range_std")
        self.code_idx_trg = self.spk_idx_dict[spk_name_trg]

        return 


    def world_feature_extract(self, wav_path):

        # load wav
        wav, fs = sf.read(wav_path)

        # check sampling frequency
        if not fs == self.cfg.dataset.sampling_rate:
            raise Exception("ERROR: sampling frequency is not matched.")

        # highpass cutoff
        highpass_cutoff = self.cfg.dataset.highpass_cutoff
        if highpass_cutoff != 0:
            wav = world_feature.low_cut_filter(wav, fs, highpass_cutoff)

        # WORLD 计算 F0 基频 / SP 频谱包络 / AP 非周期序列
        _, f0_range, spc_range, ap_range = world_feature.analyze_range(wav, fs=fs, minf0=self.minf0, \
                                                            maxf0=self.maxf0, fperiod=self.cfg.dataset.shiftms, fftl=self.cfg.dataset.fft_size)

        # 生成 uv 特征 / 连续 F0 基频
        # uv: an unvoiced/voiced binary decision feature
        uv_range, cont_f0_range = world_feature.convert_continuos_f0(np.array(f0_range))
        uv_range = np.expand_dims(uv_range, axis=-1)

        # 低通滤波器修正 F0 基频
        # F0: log-scaled of continuous F0
        cont_f0_lpf_range = world_feature.low_pass_filter(cont_f0_range, int(1.0 / (self.cfg.dataset.shiftms * 0.001)), cutoff=world_feature.LOWPASS_CUTOFF)
        log_cont_f0_lpf_range = np.expand_dims(np.log(cont_f0_lpf_range), axis=-1)

        # 编码 AP 非周期序列
        # codeap: 3-dimensional aperiodicity coding coefficien
        codeap_range = pw.code_aperiodicity(ap_range, fs)
        if codeap_range.ndim == 1:
            codeap_range = np.expand_dims(codeap_range, axis=-1)

        # 生成 mcep 倒谱系数
        # mcep：mel-cepstrum coeficient
        mcep_range = ps.sp2mc(spc_range, self.cfg.dataset.mcep_dim, self.cfg.dataset.mcep_alpha)

        # 特征拼接 uv 特征 / 对数 F0 基频 / 编码 AP 非周期序列 / mcep 倒谱系数
        feat_org_lf0 = np.c_[uv_range, log_cont_f0_lpf_range, codeap_range, mcep_range]

        return f0_range, spc_range, ap_range, uv_range, log_cont_f0_lpf_range, codeap_range, mcep_range, feat_org_lf0


    def world_feature_fuse(self, f0, codeap, mcep_cv):
        
        # 基频转换，转换到 trg 的基频上
        f0_cv = world_feature.convert_f0(f0, self.f0_range_mean_src, self.f0_range_std_src, self.f0_range_mean_trg, self.f0_range_std_trg)

         # 生成 uv 特征 / 连续 F0 基频
        uv_cv, contf0_cv = world_feature.convert_continuos_f0(np.array(f0_cv))
        uv_cv = np.expand_dims(uv_cv, axis=-1)

         # 低通滤波器修正 F0 基频
        cont_f0_lpf_cv = world_feature.low_pass_filter(contf0_cv, int(1.0 / (self.cfg.dataset.shiftms * 0.001)), cutoff=world_feature.LOWPASS_CUTOFF)
        log_cont_f0_lpf_cv = np.expand_dims(np.log(cont_f0_lpf_cv), axis=-1)

        # 特征拼接 uv 特征 / 对数 F0 基频 / 编码 AP 非周期序列 / mcep 倒谱系数
        feat_cv = np.c_[uv_cv, log_cont_f0_lpf_cv, codeap, mcep_cv]

        return feat_cv, f0_cv, uv_cv, log_cont_f0_lpf_cv


    def world_feature_2wav(self, f0, f0_cv, ap, mcep, mcep_cv):

        fs = self.cfg.dataset.sampling_rate

        sp_cv = ps.mc2sp(mcep_cv, self.cfg.dataset.mcep_alpha, self.cfg.dataset.fft_size)
        wav_cv = np.clip(pw.synthesize(f0_cv, sp_cv, ap, fs, frame_period=self.cfg.dataset.shiftms), -1, 1)

        sp = ps.mc2sp(mcep, self.cfg.dataset.mcep_alpha, self.cfg.dataset.fft_size)
        wav_anasyn = np.clip(pw.synthesize(f0, sp, ap, fs, frame_period=self.cfg.dataset.shiftms), -1, 1)

        return wav_cv, wav_anasyn


    def model_forward(self, feat):

        lat_feat_src, _, _, _, _ = \
            self.model['encoder'](torch.FloatTensor(feat).cuda(), self.y_in_pp, sampling=False)

        src_code = np.zeros((lat_feat_src.shape[0], self.n_spk))
        src_code[:, self.code_idx_src] = 1
        src_code = torch.FloatTensor(src_code).cuda()

        trg_code = np.zeros((lat_feat_src.shape[0], self.n_spk))
        trg_code[:, self.code_idx_trg] = 1
        trg_code = torch.FloatTensor(trg_code).cuda()

        cvmcep_src, _, _ = self.model['decoder'](torch.cat((src_code, lat_feat_src),1), self.y_in_src)
        cvmcep_src = np.array(cvmcep_src.cpu().data.numpy(), dtype=np.float64)

        mcep_cv, _, _ = self.model['decoder'](torch.cat((trg_code, lat_feat_src),1), self.y_in_trg)
        mcep_cv = np.array(mcep_cv.cpu().data.numpy(), dtype=np.float64)
        mcep_cv= np.ascontiguousarray(mcep_cv)

        return mcep_cv


    def voice_conversion(self, wav_path):
        
        # feature extract
        f0, sp, ap, uv, log_cont_f0_lpf, codeap, mcep, feat_src = self.world_feature_extract(wav_path)

        # model forward
        mcep_cv = self.model_forward(feat_src)

        feat_cv, f0_cv, uv_cv, log_cont_f0_lpf_cv = self.world_feature_fuse(f0, codeap, mcep_cv)

        wav_cv, wav_anasyn = self.world_feature_2wav(f0, f0_cv, ap, mcep, mcep_cv)

        return wav_cv, wav_anasyn