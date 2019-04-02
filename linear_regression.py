import torch
from matplotlib import pyplot as plt
import numpy as np

device = torch.device('cpu')

torch.manual_seed(1000) # 这样生成的随机数是固定的

def get_fake_data(batch_size=8):
    '''产生随机数据：y=x*2+3，加上一些噪声'''
    x = torch.rand(batch_size, 1, device=device) * 5 # rand():区间[0, 1)的均匀分布
    y = x * 2 + 3+ torch.randn(batch_size, 1, device=device) # randn():标准正态分布（均值为0，方差为1，即高斯白噪声）
    return x, y

# x, y = get_fake_data(batch_size=16)
# plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
# plt.show()

#随机初始化参数
# w = torch.rand(1, 1).to(device)
# b = torch.rand(1, 1).to(device)
w = torch.rand(1, 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)
losses = np.zeros(500)

lr = 0.005 #学习率

for ii in range(500):
    x, y = get_fake_data(batch_size=32)

    # forward
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2 # 均方误差
    loss = loss.sum()
    losses[ii] = loss.item()

    # backward
    # dloss = 1;
    # dy_pred = dloss * (y_pred - y)
    # dw = x.t().mm(dy_pred)
    # db = dy_pred.sum()
    loss.backward()

    # update parameters
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    # 梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()

    if ii%50 == 0:
        # display.clear_output(wait=True)
        x = torch.arange(0, 6).view(-1, 1).float()
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy()) # predict data

        x2, y2 = get_fake_data(batch_size=32)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

print('w: ', w.item(), 'b: ', b.item())

plt.plot(losses)
plt.ylim(5,50)
plt.show()