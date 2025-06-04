import torch


def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true, epsilon=10):
    return torch.mean(torch.abs((pred - true) / (true + epsilon)))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / true))


def MBE(pred, true):
    # mbd = torch.abs(torch.mean(pred - true))
    mbd = torch.mean(torch.abs(pred - true))
    mean_true = torch.mean(true)
    mbd_percentage = (mbd / mean_true)
    return mbd_percentage


def ACC(pred, true, tolerance=0.4):
    # # 过滤掉真实值为0的部分
    # non_zero_mask = true != 0
    # pred = pred[non_zero_mask]
    # true = true[non_zero_mask]
    #
    # # 计算误差
    # errors = torch.abs(pred - true)
    #
    # # 计算容忍范围
    # tolerance_range = tolerance * torch.abs(true)
    #
    # # 判断预测是否在容忍范围内
    # within_tolerance = errors <= tolerance_range
    #
    # # 计算准确率
    # accuracy = torch.mean(within_tolerance.float()) * 100
    #
    # return accuracy

    # non_zero_mask = true != 0
    # pred = pred[non_zero_mask]
    # true = true[non_zero_mask]
    # # 计算相对误差 Ei
    # mean_true = torch.mean(true)
    # Ei = torch.abs(pred - true) / mean_true * 100
    # # 计算 A1 指标
    # A1 = 1 - torch.sqrt(torch.mean(Ei ** 2)) * 100
    # return A1

    e = torch.sum(torch.abs(pred - true))
    sum_true = torch.sum(true)
    percentage = 1 - (e / sum_true)
    return percentage

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


if __name__ == "__main__":
    x = torch.rand(2, 2)
    print(x)
    y = torch.rand(2, 2)
    print(y)
    out = MAE(x, y)
    print(out.shape)
    print(out)
