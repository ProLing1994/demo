# r151-wa
# ===================================================== # 
# datasets
DATALIST = [
    '/lirui/DATA/BSDBaseCoco/BSDwideangleSegOld', 
    '/lirui/DATA/BSDDaily/LongFocus/20220425_aeb', 
    '/lirui/DATA/BSDDaily/LongFocus/20220429_aeb', 
    '/lirui/DATA/BSDDaily/LongFocus/20220512_aeb', 
    '/lirui/DATA/BSDDaily/LongFocus/20220527_aeb',
    '/lirui/DATA/BSDDaily/WideAngle/20220628_c53_WideAngle_detseg', 
    '/lirui/DATA/BSDDaily/WideAngle/20220706_c53_wideangle_detseg_fix', 
    '/lirui/DATA/BSDDaily/WideAngle/20220708_c53_WideAngle_detseg', 
    '/lirui/DATA/BSDDaily/WideAngle/20220630_c53_wideangle_detseg', 
]

# ===================================================== # 
# ClassTables
seg_classes = ['rail', 'roadside', 'green_belts', 'person',]
box_classes = []
ClassTable = ['_background_', 'rail', 'roadside', 'green_belts', 'person', ] # ensure "_background_" in position 0
# ===================================================== # 
# cfgs
cfg_modelcode = 'base_220912_32c_128'
cfg_modeltype = 'lanenet_big' # 'lanenet' 'lanenet_big'
cfg_img_c = 3
cfg_img_hw = (128, 128) #(128, 128) # (224, 224)
cfg_rgb_means = (127, 127, 127)
cfg_color_map = [[0,0,0], [0,255,255], [0,0,255], [0,255,0], [255,0,0]]
cfg_class_weight = [1, 2, 5, 1, 10]
cfg_backbone = None # 'Seg16'
cfg_type_block = None


