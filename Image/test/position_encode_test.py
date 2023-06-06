import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [np.sin(pos / np.power(10000, 2 * i / embed_dim)) if i%2==0 else 
            np.cos(pos / np.power(10000, 2*i/embed_dim))
            for i in range(embed_dim) ]
            for pos in range(max_seq_len)])
    
    return positional_encoding


positional_encoding = get_positional_encoding(max_seq_len=100, embed_dim=16)

# plt.figure(figsize=(8, 5))
# plt.plot(positional_encoding[:, 0], label="dimension 0")
# plt.plot(positional_encoding[:, 1], label="dimension 1")
# plt.plot(positional_encoding[:, 2], label="dimension 2")
# plt.plot(positional_encoding[:, 5], label="dimension 5")
# plt.plot(positional_encoding[:, 10], label="dimension 10")
# plt.plot(positional_encoding[:, 15], label="dimension 15")
# plt.legend()
# plt.xlabel("Sequence length")
# plt.ylabel("Period of Positional Encoding")
# plt.savefig('/home/huanyuan/code/demo/Image/test/test.jpg', dpi=300) 

plt.figure(figsize=(8, 5))
plt.plot(positional_encoding[0, :], label="Sequence 0")
plt.plot(positional_encoding[1, :], label="Sequence 1")
plt.plot(positional_encoding[2, :], label="Sequence 2")
plt.plot(positional_encoding[3, :], label="Sequence 3")
plt.legend()
plt.xlabel("dimension length")
plt.ylabel("Period of Positional Encoding")
plt.savefig('/home/huanyuan/code/demo/Image/test/test.jpg', dpi=300) 