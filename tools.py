import torch
from param import device, device_cpu


def my_zeros(shape):
    return torch.zeros(shape, device=device, dtype=torch.float32)


def my_ones(shape):
    return torch.ones(shape, device=device, dtype=torch.float32)


# 返回n*n的单位矩阵
def my_eye(n):
    return torch.eye(n, device=device, dtype=torch.float32)


def my_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.float32)


def my_long_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.long)

# 返回大小为 (end-start)/step的一维张量 其值介于[start,end),步长为step
def my_range(start, end, step=1):
    return torch.arange(start=start, end=end, step=step, device=device, dtype=torch.float32)


def dist_mat(x, y, inplace=True):
    # y转置然后与x相乘
    d = torch.mm(x, y.transpose(0, 1))
    # 计算每行的平方和然后在结果张量中添加一个额外的维度
    v_x = torch.sum(x ** 2, 1).unsqueeze(1)
    v_y = torch.sum(y ** 2, 1).unsqueeze(0)
    # 将变量d乘以-2，相当于将点积结果乘以-2。这一步是计算欧氏距离的平方中的负数项
    d *= -2
    if inplace:
        d += v_x
        d += v_y
    else:
        d = d + v_x
        d = d + v_y

    # 两个输入张量之间的欧氏距离的平方矩阵
    # d(i,j)表示x中第i行和y中第j行之间的欧氏距离的平方
    return d


def nn_search(y, x):
    d = dist_mat(x, y)
    return torch.argmin(d, dim=1)
