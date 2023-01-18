# schoolbus
# ===================================================== # 
# datasets
DATALIST = [

    '/lirui/DATA/SchoolBusSeg/SchoolBusSeg/base', 
]

# ===================================================== # 
# ClassTables
seg_classes = ['roadside', 'rail']
box_classes = []
ClassTable = ['_background_', 'roadside', 'rail'] # ensure "_background_" in position 0
# ===================================================== # 
# cfgs
cfg_modelcode = 'schoolbus_220924_16c_256'
cfg_modeltype = 'lanenet' # 'lanenet' 'lanenet_big'
cfg_img_c = 3
cfg_img_hw = (256, 256) #(128, 128) # (224, 224) # (256, 256)
cfg_rgb_means = (127, 127, 127)
cfg_color_map = [[0,0,0], [0,0,255], [0,255,255]]
cfg_class_weight = [1, 5, 2]
cfg_backbone = None # 'Seg16'
cfg_type_block = None


