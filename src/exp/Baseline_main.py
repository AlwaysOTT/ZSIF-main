# from src.utils.tools import adjust_learning_rate, visual, test_params_flop
# from src.utils.metrics import metric
import torch
import os
import time
import warnings
# import matplotlib.pyplot as plt
import numpy as np
import wandb
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.optim import lr_scheduler

from src.exp.exp_basic import Exp_Basic
from src.models import PatchTST, Crossformer, Autoformer, Informer, FiLM, LightTS, FEDformer, DLinear
from src.utils.metrics import MAE, RMSE, MSE
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
        # print('Use GPU: cuda:{}'.format(self.gpu))
        self._build_model()
        # if self.gpu == 0:
        #     self._build_wandb()

    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'Crossformer': Crossformer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'FiLM': FiLM,
            'LightTS': LightTS,
            'FEDformer': FEDformer,
            'DLinear': DLinear
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

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.optimizer.lr,
                                  weight_decay=self.optimizer.weight_decay, betas=self.optimizer.betas)
        return model_optim


    def prepare_batch(self, batch):
        x_ts = batch["x"].float().to(self.device, non_blocking=True)
        y_ts = batch["y"].float().to(self.device, non_blocking=True)
        x_timestamp = batch["x_timestamp"].float().to(self.device, non_blocking=True)
        y_timestamp = batch["y_timestamp"].float().to(self.device, non_blocking=True)
        ts_coords = batch["station_coords"].float().to(self.device, non_blocking=True)
        ts_elevation = batch["station_elevation"].float().to(self.device, non_blocking=True)
        ghi_id = batch["ghi"].long()[0].to(self.device, non_blocking=True)
        # print(f"ts:{x_ts.shape}")
        # print(f"ts_coords:{ts_coords.shape}")
        # print(f"ts_elevation:{ts_elevation.shape}")

        T = x_ts.shape[1]
        ts_elevation_x = ts_elevation[..., 0, 0]
        ts_elevation_x = ts_elevation_x[(...,) + (None,) * 2]
        ts_elevation_x = ts_elevation_x.repeat(1, T, 1)

        ts_coords_x = ts_coords[..., 0, 0]
        ts_coords_x = ts_coords_x.unsqueeze(1)
        ts_coords_x = ts_coords_x.repeat(1, T, 1)

        # print(f"x_ts:{x_ts.shape}")
        # print(f"ts_coords_x:{ts_coords_x.shape}")
        # print(f"ts_elevation_x:{ts_elevation_x.shape}")
        x_ts = torch.cat([x_ts, ts_coords_x, ts_elevation_x], axis=-1)

        # do the same with y_ts
        T = y_ts.shape[1]
        ts_elevation_y = ts_elevation[..., 0, 0]
        ts_elevation_y = ts_elevation_y[(...,) + (None,) * 2]
        ts_elevation_y = ts_elevation_y.repeat(1, T, 1)

        ts_coords_y = ts_coords[..., 0, 0]
        ts_coords_y = ts_coords_y.unsqueeze(1)
        ts_coords_y = ts_coords_y.repeat(1, T, 1)

        y_ts = torch.cat([y_ts, ts_coords_y, ts_elevation_y], axis=-1)

        return x_ts, y_ts, x_timestamp, y_timestamp, ghi_id

    def train(self):
        train_data, train_loader = self.data_loader.data_provider(flag='train', logger=self.logger, cfg=self.cfg)
        vali_data, vali_loader = self.data_loader.data_provider(flag='val', logger=self.logger, cfg=self.cfg)
        test_data, test_loader = self.data_loader.data_provider(flag='test', logger=self.logger, cfg=self.cfg)

        path = self.hyparams.checkpoint
        if self.gpu == 0:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        val_steps = len(vali_loader)
        test_steps = len(test_loader)
        # model_optim = self._select_optimizer()
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=model_optim,
        #                                            mode=self.scheduler.mode,
        #                                            patience=self.scheduler.patience,
        #                                            factor=self.scheduler.factor,
        #                                            verbose=self.scheduler.verbose)

        for epoch in range(self.hyparams.train_epoch):
            if self.cfg.multi_gpu:
                train_loader.sampler.set_epoch(epoch)
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, train_batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, ghi_id = self.prepare_batch(train_batch)
                iter_count += 1
                self.optimizer.zero_grad()

                if self.hyparams.padding == 0:
                    decoder_input = torch.zeros(
                        (batch_y.size(0), self.data_loader.dataset.pred_len, batch_y.size(-1))).type_as(
                        batch_y)
                else:  # self.hparams.padding == 1
                    decoder_input = torch.ones(
                        (batch_y.size(0), self.data_loader.dataset.pred_len, batch_y.size(-1))).type_as(
                        batch_y)
                decoder_input = torch.cat([batch_y[:, : self.data_loader.dataset.label_len, :], decoder_input],
                                          dim=1)
                # encoder - decoder
                if self.hyparams.use_amp:
                    with amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
                        if self.hyparams.output_attention:
                            outputs = outputs[0]
                        outputs = outputs[:, -self.hyparams.pred_len:,
                                  ghi_id]  # 0 is the index of the target variable GHI , 8 is the index of the target variable DHI
                        batch_y = batch_y[:, -self.hyparams.pred_len:, ghi_id]
                        loss = self.criterion(outputs, batch_y)

                else:
                    outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
                    if self.hyparams.output_attention:
                        outputs = outputs[0]
                    outputs = outputs[:, -self.hyparams.pred_len:,
                              ghi_id]  # 0 is the index of the target variable GHI , 8 is the index of the target variable DHI
                    batch_y = batch_y[:, -self.hyparams.pred_len:, ghi_id]
                    loss = self.criterion(outputs, batch_y)
                if self.cfg.multi_gpu:
                    loss_value = all_reduce_mean(loss).item()
                else:
                    loss_value = loss.item()
                train_loss.append(loss_value)

                if self.gpu == 0 and (i + 1) % 200 == 0:
                    # print(f"ts_x:{batch_x}")
                    # print(f"outputs:{outputs}")
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

            train_loss = np.average(train_loss)
            val_loss = self.val(vali_data, vali_loader)
            # test_loss = self.val(test_data, test_loader, criterion)
            self.scheduler.step(val_loss["mae"])
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
            # self.early_stop(val_loss["mae"], self.model, path, self.gpu)
            if self.early_stop.early_stop:
                if self.gpu == 0:
                    self.logger.info("Early stopping")
                break

            # adjust_learning_rate(model_optim, schduler, epoch + 1, self.hyparams.lradj, self.hyparams.lr)

        # wandb.finish()
        return self.model

    def val(self, val_data, val_loader):
        mse_loss = []
        mae_loss = []
        rmse_loss = []
        out_all = []
        y_all = []
        self.model.eval()
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, ghi_id = self.prepare_batch(val_batch)
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                if self.hyparams.padding == 0:
                    decoder_input = torch.zeros(
                        (batch_y.size(0), self.data_loader.dataset.pred_len, batch_y.size(-1))).type_as(
                        batch_y)
                else:  # self.hparams.padding == 1
                    decoder_input = torch.ones(
                        (batch_y.size(0), self.data_loader.dataset.pred_len, batch_y.size(-1))).type_as(
                        batch_y)
                decoder_input = torch.cat([batch_y[:, : self.data_loader.dataset.label_len, :], decoder_input],
                                          dim=1).float()
                # encoder - decoder
                if self.hyparams.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
                        if self.hyparams.output_attention:
                            outputs = outputs[0]
                        outputs = outputs[:, -self.hyparams.pred_len:,
                                  ghi_id]  # 0 is the index of the target variable GHI , 8 is the index of the target variable DHI
                        batch_y = batch_y[:, -self.hyparams.pred_len:, ghi_id]
                        # loss = self.criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
                    if self.hyparams.output_attention:
                        outputs = outputs[0]
                    outputs = outputs[:, -self.hyparams.pred_len:,
                              ghi_id]  # 0 is the index of the target variable GHI , 8 is the index of the target variable DHI
                    batch_y = batch_y[:, -self.hyparams.pred_len:, ghi_id]
                    # loss = self.criterion(outputs, batch_y)
                out_all.append(outputs)
                y_all.append(batch_y)
                # loss_value = all_reduce_mean(loss).item()
                # mse_loss.append(loss_value)
                # mae_l = MAE(outputs, batch_y)
                # mae_loss.append(mae_l.cpu())
                # rmse_l = RMSE(outputs, batch_y)
                # rmse_loss.append(rmse_l.cpu())
            # mse = MSE(torch.concat(out_all, dim=0), torch.concat(y_all, dim=0))
            rmse = RMSE(torch.concat(out_all, dim=0), torch.concat(y_all, dim=0))
            mae = MAE(torch.concat(out_all, dim=0), torch.concat(y_all, dim=0))
            # mse = all_reduce_mean(mse)
            rmse = all_reduce_mean(rmse)
            mae = all_reduce_mean(mae)
        # mse_loss = np.average(mse_loss)
        # mae_loss = np.average(mae_loss)
        # rmse_loss = np.average(rmse_loss)
        total_loss = {"mse": mse_loss, "mae": mae, "rmse": rmse}
        self.model.train()
        return total_loss


    def test(self):
        test_data, test_loader = self.data_loader.data_provider(flag='test', logger=self.logger, cfg=self.cfg)

        self.logger.info('Testing loading model...')
        self.model.load_state_dict(
            torch.load(os.path.join(self.hyparams.test_resume, 'best.pth'), map_location='cpu')['model'])
        time_now = time.time()
        test_loss = self.val(test_data, test_loader)
        self.logger.info("cost time: {}".format(time.time() - time_now))
        if self.gpu == 0:
            # wandb.log({"test_mae": test_loss["mae"]})
            # wandb.log({"test_rmse": test_loss["rmse"]})
            self.logger.info(
                "Test mae:{0:.5f} , Test rmse:{1:.5f}".format(
                    test_loss["mae"].float(), test_loss["rmse"].float()))
