from einops import rearrange, repeat
import torch.nn.functional as F
from torch import optim
import wandb
from torch.cuda import amp

from src.exp.exp_basic import Exp_Basic
# from src.optim.cosine_warmup import CosineWarmupScheduler
# from src.robust_loss_pytorch import AdaptiveLossFunction
from torch.optim.lr_scheduler import LambdaLR

from src.losses.quantile_loss import QuantileLossMulti
from src.models import ZSIF
from src.optim.cosine_warmup import CosineWarmupScheduler
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.metrics import MAE, RMSE, MAPE, ACC, MBE
import torch
import os
import time
import warnings
# import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch_npu.npu import amp

from src.utils.tools import resume_model, visual
from src.utils.torch_dist import all_reduce_mean

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self,
                 name: str,
                 dataloader,
                 # model: torch.nn.Module,
                 model: dict,
                 optimizer,
                 early_stop,
                 scheduler,
                 criterion: torch.nn.Module,
                 hyparams: dict,
                 wandb_para: dict
                 ):
        super().__init__()
        self.name = name
        self.model_para = model
        self.data_loader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stop = early_stop
        self.scheduler = scheduler
        self.hyparams = hyparams
        self.wandb_para = wandb_para
        self.scaler = None

    def acquire_device(self, gpu, cfg):
        self.gpu = gpu
        self.cfg = cfg
        torch.cuda.set_device(gpu)
        self.device = torch.device('cuda:{}'.format(gpu))
        self._build_model()

    def _build_model(self):
        model_dict = {
            'ZSIF': ZSIF
        }
        self.model = model_dict[self.name].Model(**self.model_para).float()
        self.model = self.model.to(self.device)
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info('number of params: {} M'.format(n_parameters / 1e6))
        if self.cfg.multi_gpu:
            self.model = DDP(self.model, device_ids=[self.gpu], find_unused_parameters=False)
        # self.criterion = AdaptiveLossFunction(48, torch.float32, self.device, alpha_hi=3.0)
        # self.criterion_opt = optim.AdamW(list(self.criterion.parameters()), lr=0.001)
        self.optimizer = self._select_optimizer()
        self.scheduler = self.scheduler(self.optimizer)
        # self.scheduler = CosineWarmupScheduler(self.optimizer, self.scheduler.warmup, self.scheduler.max_iters,
        #                                        self.scheduler.verbose)
        # lr_lambda = lambda epoch: (0.7 ** epoch)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda, verbose=True)
        # if self.gpu == 0:
        #     wandb.init(
        #         project=self.wandb_para.project,
        #         name=self.wandb_para.name,
        #         dir=self.wandb_para.dir,
        #         mode=self.wandb_para.mode,
        #         group=self.wandb_para.group
        #     )
        if self.hyparams.use_amp:
            self.scaler = amp.GradScaler()

        self.quantile_loss = QuantileLossMulti()

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.optimizer.lr,
                                  weight_decay=self.optimizer.weight_decay, betas=self.optimizer.betas)
        return model_optim

    def calc_dot_product(
            self,
            optflow_data: torch.Tensor,
            coords: torch.Tensor,
            station_coords: torch.Tensor,
    ):
        """
        Args:
            optflow_data: Optical flow data. Tensor of shape [B, T, 2*C, H, W]
            coords: Coordinates of each pixel. Tensor of shape [B, 2, H, W]
            station_coords: Coordinates of the station. Tensor of shape [B, 2, 1, 1]
        Returns:
            dp: Dot product between the optical flow and station vectors of shape [B, T, C, H, W]
        """
        optflow_data = rearrange(optflow_data, "b t (c n) h w -> b t c n h w", n=2)
        # 为什么要减去图片坐标呢
        dist = station_coords - coords
        dist = repeat(
            dist,
            "b n h w -> b t c n h w",
            t=optflow_data.shape[1],
            c=optflow_data.shape[2],
        )
        optflow_data = F.normalize(
            rearrange(optflow_data, "b t c n h w -> b t c h w n"), p=2, dim=-1
        )
        dist = F.normalize(rearrange(dist, "b t c n h w -> b t c h w n"), p=2, dim=-1)
        dp = optflow_data[..., 0] * dist[..., 0] + optflow_data[..., 1] * dist[..., 1]
        return dp

    # def prepare_batch(self, batch, use_target=True, use_DCT=True):
    #     x_ctx = batch["context"].float().to(self.device, non_blocking=True)
    #     x_opt = batch["optical_flow"].float().to(self.device, non_blocking=True)
    #     x_ts = batch["timeseries"].float().to(self.device, non_blocking=True)
    #     ts_mean = batch["ts_mean"].float().to(self.device, non_blocking=True)
    #     ts_std = batch["ts_std"].float().to(self.device, non_blocking=True)
    #     if use_target:
    #         y_ts = batch["target"].float().to(self.device, non_blocking=True)
    #         y_previous_ts = batch["target_previous"].float().numpy()
    #     # if use_DCT:
    #     #     x_ts = x_ts.permute(0, 2, 1)
    #     #     x_ts = self.dct_layer(x_ts)
    #     #     x_ts = x_ts.permute(0, 2, 1)
    #     ctx_coords = batch["spatial_coordinates"].float().to(self.device, non_blocking=True)
    #     time_coords = batch["time_coordinates"].float().to(self.device, non_blocking=True)
    #     ts_coords = batch["station_coords"].float().to(self.device, non_blocking=True)
    #     ts_elevation = batch["station_elevation"].float().to(self.device, non_blocking=True)
    #
    #     aux_data = []
    #     for k in batch["auxiliary_data"]:
    #         ad = batch["auxiliary_data"][k].float().to(self.device, non_blocking=True)  # B, H, W
    #         ad = ad.unsqueeze(1).unsqueeze(1)
    #         ad = ad.repeat(1, x_ctx.shape[1], 1, 1, 1)
    #         aux_data.append(ad)
    #     # x_ctx = torch.cat(
    #     #     [
    #     #         x_ctx,
    #     #     ]
    #     #     + aux_data,
    #     #     axis=2,
    #     # )
    #
    #     # if self.hyparams.use_dp:
    #     #     x_dp = self.calc_dot_product(x_opt, spatial_coords, ts_coords)
    #     #     x_ctx = torch.cat([x_ctx, x_dp], axis=2)
    #     # else:
    #     #     x_ctx = torch.cat([x_ctx, x_opt], axis=2)
    #     ts_elevation = ts_elevation[..., 0, 0]
    #     ts_elevation = ts_elevation[(...,) + (None,) * 2]
    #     ts_elevation = ts_elevation.repeat(1, x_ts.shape[1], 1)
    #     x_ts = torch.cat([x_ts, ts_elevation], axis=-1)
    #
    #     # H, W = x_ctx.shape[-2:]
    #     # ctx_coords = F.interpolate(
    #     #     spatial_coords,
    #     #     size=(H // self.hyparams.patch_size[0], W // self.hyparams.patch_size[1]),
    #     #     mode="bilinear",
    #     # )
    #     if use_target:
    #         return x_ts, x_ctx, y_ts, y_previous_ts, ctx_coords, ts_coords, time_coords, ts_mean, ts_std
    #     return x_ts, x_ctx, ctx_coords, ts_coords, time_coords

    def prepare_batch(self, batch, use_target=True, use_DCT=True):
        x_ctx = batch["context"].float().to(self.device)
        # x_opt = batch["optical_flow"].float().to(self.device, non_blocking=True)
        ctx_coords = batch["context_coordinates"].float().to(self.device)
        # ctx_elevation = batch["context_elevation"].float().to(self.device, non_blocking=True)

        x_ts = batch["timeseries"].float().to(self.device)
        time_coords = batch["time_embedding"].float().to(self.device)
        ts_coords = batch["station_coordinates"].float().to(self.device)
        ts_elevation = batch["station_elevation"].float().to(self.device)

        ts_mean = batch["ts_mean"].float().to(self.device)
        ts_std = batch["ts_std"].float().to(self.device)
        if use_target:
            y_ts = batch["target"].float().to(self.device)
            y_previous_ts = batch["target_previous"].float().numpy()
        # if use_DCT:
        #     x_ts = x_ts.permute(0, 2, 1)
        #     x_ts = self.dct_layer(x_ts)
        #     x_ts = x_ts.permute(0, 2, 1)

        # aux_data = []
        # for k in batch["auxiliary_data"]:
        #     ad = batch["auxiliary_data"][k].float().to(self.device, non_blocking=True)  # B, H, W
        #     ad = ad.unsqueeze(1).unsqueeze(1)
        #     ad = ad.repeat(1, x_ctx.shape[1], 1, 1, 1)
        #     aux_data.append(ad
        # x_ctx = torch.cat([x_ctx, ctx_elevation], axis=2)  # 拼接上高度特征

        # if self.hyparams.use_dp:
        #     x_dp = self.calc_dot_product(x_opt, spatial_coords, ts_coords)
        #     x_ctx = torch.cat([x_ctx, x_dp], axis=2)
        # else:
        #     x_ctx = torch.cat([x_ctx, x_opt], axis=2)
        # ts_elevation = ts_elevation[..., 0, 0]
        # ts_elevation = ts_elevation[(...,) + (None,) * 2]
        # ts_elevation = ts_elevation.repeat(1, x_ts.shape[1], 1)
        x_ts = torch.cat([x_ts, ts_elevation], axis=-1)

        # H, W = x_ctx.shape[-2:]
        # ctx_coords = F.interpolate(
        #     ctx_coords,
        #     size=(H // self.hyparams.patch_size[0], W // self.hyparams.patch_size[1]),
        #     mode="bilinear",
        # )

        return x_ts, x_ctx, y_ts, y_previous_ts, ctx_coords, ts_coords, time_coords, ts_mean, ts_std

    def train(self):
        train_data, train_loader = self.data_loader.data_provider(flag='train', logger=self.logger, cfg=self.cfg)
        val_data, val_loader = self.data_loader.data_provider(flag='val', logger=self.logger, cfg=self.cfg)
        test_data, test_loader = self.data_loader.data_provider(flag='test', logger=self.logger, cfg=self.cfg)

        path = self.hyparams.checkpoint
        if self.gpu == 0:
            if not os.path.exists(path):
                os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        val_steps = len(val_loader)
        test_steps = len(test_loader)

        start_epoch = 0
        if self.hyparams.resume:
            start_epoch = resume_model(self.logger, self.model, self.hyparams.resume_ck, self.optimizer, self.scaler,
                                       self.scheduler)
        for epoch in range(start_epoch, self.hyparams.train_epoch):
            if self.cfg.multi_gpu:
                train_loader.sampler.set_epoch(epoch)
            # print(model_optim.state_dict()['param_groups'][0]['lr'])
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, train_batch in enumerate(train_loader):
                (x_ts, x_ctx, y_ts, y_prev_ts, ctx_coords, ts_coords, time_coords, ts_mean,
                 ts_std) = self.prepare_batch(train_batch)

                iter_count += 1
                self.optimizer.zero_grad()
                if self.hyparams.use_amp:
                    with amp.autocast():
                        out, self_attention_scores, cross_attention_scores = self.model(
                            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
                        )
                        # y_hat = out.mean(dim=2)
                        # loss = self.criterion(y_hat, y_ts)
                        loss = self.criterion(out, y_ts[:, :, 0])
                else:
                    out, self_attention_scores, cross_attention_scores = self.model(
                        x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
                    )
                    # y_hat = out.mean(dim=2)
                    # loss = self.criterion(y_hat, y_ts)
                    loss = self.criterion(out, y_ts[:, :, 0])
                if self.cfg.multi_gpu:
                    loss_value = all_reduce_mean(loss).item()
                else:
                    loss_value = loss.item()
                train_loss.append(loss_value)
                # if self.gpu == 0:
                #     wandb.log({"train_loss": loss_value})
                if self.gpu == 0 and (i + 1) % 200 == 0:
                    # print(f"ts_x:{x_ts}")
                    # print(f"outputs:{out}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                                (self.hyparams.train_epoch - epoch) * (train_steps + val_steps + test_steps) - i - 1)
                    self.logger.info(
                        "\tepoch: {0}->its: {1}  | loss: {2:.5f}---speed: {3:.3f}s/it; left time: {4:.3f}s".format(
                            epoch, i + 1, loss_value, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.hyparams.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                torch.cuda.synchronize()

            train_loss = np.average(train_loss)
            val_loss = self.val(val_loader)
            # test_loss = self.val(test_loader)
            self.scheduler.step(epoch)
            # adjust_learning_rate(self.optimizer, epoch + 1, "type1", self.optimizer_p.lr, True)
            if self.gpu == 0:
                # wandb.log({"train_loss_e": train_loss})
                # wandb.log({"val_mae": val_loss["mae"]})
                # wandb.log({"val_rmse": val_loss["rmse"]})
                # wandb.log({"test_mae": test_loss["mae"]})
                # wandb.log({"test_rmse": test_loss["rmse"]})
                self.logger.info("Epoch: {}, cost time: {}".format(epoch, time.time() - epoch_time))
                self.logger.info(
                    "Epoch: {0}, Steps: {1} | Train Loss:{2:.5f}, Val Loss:{3:.5f}".format(
                        epoch, train_steps, train_loss, val_loss["mae"]))
                if self.cfg.multi_gpu:
                    self.early_stop(self.logger, val_loss["mae"], self.model.module, path, epoch, self.optimizer,
                                    self.scaler, self.cfg, self.scheduler)
                else:
                    self.early_stop(self.logger, val_loss["mae"], self.model, path, epoch, self.optimizer,
                                    self.scaler, self.cfg, self.scheduler)
            if self.early_stop.early_stop:
                if self.gpu == 0:
                    self.logger.info("Early stopping")
                break
        wandb.finish()
        return self.model

    def val(self, val_loader):
        mse_loss = []
        mae_loss = []
        rmse_loss = []
        out_all = []
        y_all = []
        self.model.eval()
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                (x_ts, x_ctx, y_ts, y_prev_ts, ctx_coords, ts_coords, time_coords, ts_mean,
                 ts_std) = self.prepare_batch(val_batch)
                if self.hyparams.use_amp:
                    with torch.cuda.amp.autocast():
                        out, self_attention_scores, cross_attention_scores = self.model(
                            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
                        )
                        # y_hat = out.mean(dim=2)
                        # loss = self.criterion(y_hat, y_ts)
                        # loss = self.criterion(out, y_ts[:, :, 0])

                else:
                    out, self_attention_scores, cross_attention_scores = self.model(
                        x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
                    )
                    # y_hat = out.mean(dim=2)
                    # loss = self.criterion(y_hat, y_ts)
                    # loss = self.criterion(out, y_ts[:, :, 0])
                # mse_loss.append(loss.item())
                # out = out * ts_std[:,0,:] + ts_mean[:,0,:]
                # y_ts = y_ts[:, :, 0] * ts_std[:,0,:] + ts_mean[:,0,:]
                # rmse_l = RMSE(out, y_ts)
                # mae_l = MAE(out, y_ts)
                out_all.append(out)
                y_all.append(y_ts[:, :, 0])
                # rmse_l = RMSE(out, y_ts[:, :, 0])
                # rmse_loss.append(rmse_l.cpu())
                # mae_l = MAE(out, y_ts[:, :, 0])
                # mae_loss.append(mae_l.cpu())
            # out_all = distributed_concat(torch.concat(out_all, dim=0), len(val_loader.dataset))
            # y_all = distributed_concat(torch.concat(y_all, dim=0), len(val_loader.dataset))
            rmse = RMSE(torch.concat(out_all, dim=0), torch.concat(y_all, dim=0))
            mae = MAE(torch.concat(out_all, dim=0), torch.concat(y_all, dim=0))
            acc = ACC(torch.concat(out_all, dim=0), torch.concat(y_all, dim=0))
            # mbe = MBE(torch.concat(out_all, dim=0), torch.concat(y_all, dim=0))
            rmse = all_reduce_mean(rmse)
            mae = all_reduce_mean(mae)
            acc = all_reduce_mean(acc)
            # mbe = all_reduce_mean(mbe)
        # mse_loss = np.average(mse_loss)
        # mae_loss = np.average(mae_loss)
        # rmse_loss = np.average(rmse_loss)
        # rmse = all_reduce_mean(rmse_loss)
        # mae = all_reduce_mean(mae_loss)
        total_loss = {"mse": mse_loss, "mae": mae, "rmse": rmse, "acc": acc}
        self.model.train()
        return total_loss

    # def test(self):
    #     test_data, test_loader = self.data_loader.data_provider(flag='test', logger=self.logger)
    #
    #     self.logger.info('Testing loading model...')
    #     self.model.module.load_state_dict(
    #         torch.load(os.path.join(self.hyparams.test_resume, 'best.pth'), map_location='cpu')['model'])
    #
    #     preds = []
    #     trues = []
    #     inputx = []
    #     mae_loss = []
    #     rmse_loss = []
    #     folder_path = self.hyparams.test_result
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, test_batch in enumerate(test_loader):
    #             (x_ts, x_ctx, y_ts, y_prev_ts, ctx_coords, ts_coords, time_coords, ts_mean,
    #              ts_std) = self.prepare_batch(test_batch)
    #             if self.hyparams.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     out, self_attention_scores, cross_attention_scores = self.model(
    #                         x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
    #                     )
    #                     y_hat = out.mean(dim=2)
    #                     loss = self.criterion(y_hat, y_ts)
    #                     mae_loss.append(loss.item())
    #                     rmse_l = RMSE(y_hat, y_ts)
    #                     rmse_loss.append(rmse_l.cpu())
    #             else:
    #                 out, self_attention_scores, cross_attention_scores = self.model(
    #                     x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
    #                 )
    #                 y_hat = out.mean(dim=2)
    #                 loss = self.criterion(y_hat, y_ts)
    #                 mae_loss.append(loss.item())
    #                 rmse_l = RMSE(y_hat, y_ts)
    #                 rmse_loss.append(rmse_l.cpu())
    #             pred = y_hat.detach().cpu().numpy()
    #             true = y_ts.detach().cpu().numpy()
    #             preds.append(pred)
    #             trues.append(true)
    #             inputx.append(y_prev_ts)
    #             if i % 50 == 0:
    #                 gt = np.concatenate((y_prev_ts[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((y_prev_ts[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    #
    #     mae_loss = np.average(mae_loss)
    #     rmse_loss = np.average(rmse_loss)
    #     self.logger.info("Test mae_loss:{0:.5f}, Test rmse_loss:{1:.5f}".format(
    #         mae_loss, rmse_loss))

    def test(self):
        test_data, test_loader = self.data_loader.data_provider(flag='test', logger=self.logger, cfg=self.cfg)

        self.logger.info('Testing loading model...')
        self.model.module.load_state_dict(
            torch.load(os.path.join(self.hyparams.test_resume, 'best.pth'), map_location='cpu')['model'])
        time_now = time.time()
        test_loss = self.val(test_loader)
        self.logger.info("cost time: {}".format(time.time() - time_now))
        if self.gpu == 0:
            # wandb.log({"test_mae": test_loss["mae"]})
            # wandb.log({"test_rmse": test_loss["rmse"]})
            self.logger.info(
                "Test mae:{0:.5f} , Test rmse:{1:.5f}, Test acc:{2:.5f}".format(
                    test_loss["mae"].float(), test_loss["rmse"].float(), test_loss["acc"].float()))

    # def test(self):
    #
    #     test_data, test_loader = self.data_loader.data_provider(flag='test', logger=self.logger, cfg=self.cfg)
    #     self.logger.info('Testing loading model...')
    #     self.model.load_state_dict(
    #         torch.load(os.path.join(self.hyparams.test_resume, 'best.pth'), map_location='cpu')['model'])
    #
    #     folder_path = self.hyparams.test_result
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     preds = []
    #     trues = []
    #     inputx = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         with PdfPages(os.path.join(folder_path, 'all.pdf')) as pdf:
    #             for i, val_batch in enumerate(test_loader):
    #                 (x_ts, x_ctx, y_ts, y_prev_ts, ctx_coords, ts_coords, time_coords, ts_mean,
    #                  ts_std) = self.prepare_batch(val_batch)
    #                 if self.hyparams.use_amp:
    #                     with torch.cuda.amp.autocast():
    #                         out, self_attention_scores, cross_attention_scores = self.model(
    #                             x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
    #                         )
    #                 else:
    #                     out, self_attention_scores, cross_attention_scores = self.model(
    #                         x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
    #                     )
    #                 pred = out.detach().cpu().numpy()
    #                 true = y_ts[:, :, 0].detach().cpu().numpy()
    #                 input = y_prev_ts[:, :, 0]
    #                 preds.append(pred)
    #                 trues.append(true)
    #                 inputx.append(input)
    #                 self.test_visdual(pdf, pred, true, input)

    def test_visdual(self, pdf, pred, true, input):
        # pred_array = np.concatenate(pred, axis=0)
        # flat_pred = pred_array.reshape(-1)
        # self.logger.info(f"pred:{flat_pred.size}")
        # true_array = np.concatenate(true, axis=0)
        # flat_true = true_array.reshape(-1)
        # self.logger.info(f"pred:{flat_true.size}")

        # segment_size = 1000
        # num_segments = len(flat_pred) // segment_size
        # for i in range(num_segments):
        #     start_index = i * segment_size
        #     end_index = start_index + segment_size
        #     fig = plt.figure()  # 创建一个新的图形
        #     plt.plot(flat_true[start_index:end_index], label='True', linewidth=2)
        #     plt.plot(flat_pred[start_index:end_index], label='Prediction', linewidth=1)
        #     plt.title(f'Segment {i + 1}')  # 给图形加上标题
        #     plt.legend()
        #     pdf.savefig(fig)
        #     plt.close(fig)
        # fig = plt.figure()
        # plt.plot(flat_true[56579:57191], label='GroundTruth', linewidth=2)
        # plt.plot(flat_pred[56579:57191], label='Prediction', linewidth=1)
        # plt.legend()
        # pdf.savefig(fig)  # 将图形保存到PDF中
        # plt.close(fig)  # 关闭图形以节省内存

        fig = plt.figure()  # 创建一个新的图形
        gt = np.concatenate((input[0, :], true[0, :]), axis=0)
        pd = np.concatenate((input[0, :], pred[0, :]), axis=0)
        plt.plot(pd, label='Prediction', linewidth=2)
        plt.plot(gt, label='GroundTruth', linewidth=2)
        plt.legend()
        pdf.savefig(fig)  # 将图形保存到PDF中
        plt.close(fig)  # 关闭图形以节省内存

        # for j in range(64):
        #     fig = plt.figure()  # 创建一个新的图形
        #     gt = np.concatenate((input[j, :], true[j, :]), axis=0)
        #     pd = np.concatenate((input[j, :], pred[j, :]), axis=0)
        #     plt.plot(pd, label='Prediction', linewidth=2)
        #     plt.plot(gt, label='GroundTruth', linewidth=2)
        #     # plt.title('Plot number: {}'.format(i))
        #     plt.legend()
        #     pdf.savefig(fig)  # 将图形保存到PDF中
        #     plt.close(fig)  # 关闭图形以节省内存
