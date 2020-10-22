关键词检索训练引擎
数据集：Google 发布的数据 speech_commands_v0.02.tar.gz

Speech/KWS/scripts/dataset/data_split.py 脚本生成训练、验证、测试数据，通过哈希值方式划分数据，保证相同说话人处于相同测试集
Speech/KWS/scripts/dataset/data_preload_audio.py 脚本读取音频数据，将音频压缩到 [-1, 1] 之间，同时保存采样数据，在模型训练过程中直接读取

Speech/KWS/train.py 脚本训练模型