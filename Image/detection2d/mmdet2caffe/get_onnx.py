import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import sys
sys.path.append(".")
import torch
from mmcv import Config, DictAction
from mmdet.core.export import build_model_from_cfg
from mmdet.models import build_detector

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

#python ./tools/adas/get_onnx.py configs/adas/fcos/rm_fcos.py G:\output\mmdet\work_dirs\rm_fcos\epoch_12.pth

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config',help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[320, 320],
        help='input image size')
    parser.add_argument(
        '--save_path',
        type=str,
        default='/home/huanyuan/code/demo/Image/detection2d/mmdet2caffe/onnx_model/',
        help='the path for saving onnx file ')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_model_from_cfg(args.config, args.checkpoint,
                                 args.cfg_options)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    save_name = args.config.replace('\\','/').split('/')[-1].split('.')[0]
    onnx_save_name=os.path.join(args.save_path,"{}.onnx".format(save_name))
    input_tensor = torch.randn(input_shape).cuda()
    print ("Exporting to ONNX: ", onnx_save_name)
    torch_onnx_out = torch.onnx.export(model, input_tensor, onnx_save_name,
                        export_params=True,
                        verbose=True,
                        input_names=['data'],
                        output_names=["synthesized"],
                        opset_version=11)

if __name__ == '__main__':
    main()
