# WAVERNN / VOCODER --------------------------------------------------------------------------------
voc_upsample_factors = (5, 5, 8)    # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider 
                                    # than input length