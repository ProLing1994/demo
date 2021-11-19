################################
# Experiment Parameters        #
################################
fp16_run=False


################################
# Model Parameters             #
################################
symbols_embedding_dim=512

# Encoder parameters
encoder_kernel_size=5
encoder_n_convolutions=3
encoder_embedding_dim=512

# Decoder parameters
decoder_rnn_dim=1024
prenet_dim=256
max_decoder_steps=1000
gate_threshold=0.5
p_attention_dropout=0.1
p_decoder_dropout=0.1

# Attention parameters
attention_rnn_dim=1024
attention_dim=128

# Location Layer parameters
attention_location_n_filters=32
attention_location_kernel_size=31

# Mel-post processing network parameters
postnet_embedding_dim=512
postnet_kernel_size=5
postnet_n_convolutions=5

################################
# Optimization Hyperparameters #
################################
grad_clip_thresh=1.0
mask_padding=True  # set model's padded outputs to padded values