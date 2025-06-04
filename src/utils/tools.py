import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# import torch_npu
from torch.backends import cudnn

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, lradj, learning_rate, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'type3':
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif lradj == 'constant':
        lr_adjust = {epoch: learning_rate}
    elif lradj == '3':
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate * 0.1}
    elif lradj == '4':
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate * 0.1}
    elif lradj == '5':
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate * 0.1}
    elif lradj == '6':
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate * 0.1}
    elif lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, logger, val_loss, model, path, epoch, optimizer, scaler, args, scheduler):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(logger, val_loss, model, path, epoch, optimizer, scaler, args, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(logger, val_loss, model, path, epoch, optimizer, scaler, args, scheduler)
            self.counter = 0
        # self.save_last_checkpoint(logger, model, path, epoch, optimizer, scaler, args, scheduler)

    def save_checkpoint(self, logger, val_loss, model, path, epoch, optimizer, scaler, args, scheduler):
        if self.verbose:
            logger.info(
                f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  Saving epoch-{epoch} model ...')
        self.val_loss_min = val_loss

        checkpoint_path = f"{path}/best.pth"
        if scaler is not None:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                'args': args,
            }
        else:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                "scheduler": scheduler.state_dict(),
                'args': args,
            }
        torch.save(to_save, checkpoint_path)

    def save_last_checkpoint(self, logger, model, path, epoch, optimizer, scaler, args, scheduler):
        logger.info(f'Saving last epoch-{epoch} model ...')
        checkpoint_path = f"{path}/last.pth"
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if scaler is not None:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                'args': args,
            }
        else:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                "scheduler": scheduler.state_dict(),
                'args': args,
            }
        torch.save(to_save, checkpoint_path)


def resume_model(logger, model, path, optimizer, scaler, scheduler):
    if path is not None:
        logger.info("Auto resume checkpoint: %s" % path)
        checkpoint = torch.load(path, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
    return epoch


def my_load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_flop(model, x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def random_seed(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    # os.environ['PYTHONHASHSEED'] = str(seed + rank)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def dispatch_clip_grad(parameters, value: float, mode: str = "norm", norm_type: float = 2.0):
    """Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value'
        norm_type (float): p-norm, default 2.0
    """
    if mode == "norm":
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == "value":
        torch.nn.utils.clip_grad_value_(parameters, value)
    else:
        assert False, f"Unknown clip mode ({mode})."
