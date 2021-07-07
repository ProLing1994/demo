
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision.models import resnet18
import matplotlib.pyplot as plt

model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
mode='cosineAnnWarm'
max_epoch = 4000
batch_iters = 1

if mode=='cosineAnn':
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
elif mode=='cosineAnnWarm':
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    '''
    以 T_0=5, T_mult=1 为例:
    T_0: 学习率第一次回到初始值的 epoch 位置.
    T_mult: 这个控制了学习率回升的速度
        - 如果 T_mult = 1, 则学习率在 T_0, 2*T_0, 3*T_0, ...., i*T_0, ....处回到最大值(初始学习率)
            - 5, 10, 15, 20, 25,.......处回到最大值
        - 如果 T_mult > 1, 则学习率在 T_0, (1+T_mult)*T_0, (1+T_mult+T_mult**2)*T_0, ....., (1+T_mult+T_mult**2+...+T_0**i)*T0, ....处回到最大值
            - 5, 15, 35, 75, 155,.......处回到最大值
    example:
        T_0=2, T_mult=2
    '''

plt.figure()
cur_lr_list = [] 
for epoch in range(max_epoch):
    print('epoch_{}: '.format(epoch))
    for batch in range(batch_iters):
        # scheduler.step(epoch + batch / batch_iters)
        scheduler.step(epoch)
        optimizer.step()
        #scheduler.step()
        cur_lr = optimizer.param_groups[-1]['lr']
        cur_lr_list.append(cur_lr)
        print('cur_lr: ',cur_lr)
    print('epoch_{}_end'.format(epoch))

x_list = list(range(len(cur_lr_list)))
plt.plot(x_list, cur_lr_list)
plt.show()