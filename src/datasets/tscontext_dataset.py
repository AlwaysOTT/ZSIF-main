import json
import os
import random
from typing import Dict, List, Tuple, Union
import deeplake
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import Dataset
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
from src.datasets.utils import calculate_possible_starts
from src.utils.time_features import time_features

"""
This prevents the following error occuring from the interaction between deeplake and wandb:
wandb.errors.UsageError: problem
"""
deeplake.constants.WANDB_INTEGRATION_ENABLED = False


class TSContextDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            stats_path: str,
            context_channels: Union[Tuple[str], List[str]],
            optflow_channels: Union[Tuple[str], List[str]],
            ts_channels: Union[Tuple[str], List[str]],
            ts_target_channels: Union[Tuple[str], List[str]],
            years: Dict[str, Union[int, List[str]]],
            stations: Dict[str, List[str]],
            mode: str = "train",
            use_target: bool = True,
            image_size: Tuple[int] = None,
            crop: Tuple[int] = None,
            seq_len: int = 24 * 2,
            label_len: int = 12,
            pred_len: int = 24 * 2,
    ) -> None:
        """
        Parameters
        ----------
        data_dir : str
            Absolute directory path where the deeplake dataset is located.
        stats_path : str
            Absolute directory path where the stats json file is located.
        context_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the context as input. If ``None`` is given,
            then it takes all available channels.
        optflow_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the optical flow as input. If ``None`` is given,
            then it takes all available channels.
        ts_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the timeseries as input. If ``None`` is given,
            then it takes all available channels.
        ts_target_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the timeseries as output. If ``None`` is given,
            then it takes all available channels.
        years: Dict[str, Union[int, str]]
            Dictionary containing for each mode, which years to select.
        stations: Dict[str, str]
            Dictionary containing for each mode, which stations to select.
        mode : str
            Indicates which dataset is it: train, val or test.
        use_target: bool
            Indicates whether to output target or not. In case it's not output, the input timeseries will be 2*n_steps long.
        image_size : Tuple[int]
            Interpolate to desired image size. If set to None, no interpolation is done
            and the size is kept as is.
        crop: Tuple[int]
            If not None, ``crop`` is expected to be in the following format:
                (lat_upper_left, lon_upper_left, lat_bottom_right, lon_bottom_right)
            And the context is cropped to the given parameters. Note that the context is first resized
            using ``image_size`` argument and then cropped.
        seq_len: int
            Number of frames in the input sequence.
        label_len: int
            Number of frames in the label sequence.
        pred_len: int
            Number of frames in the prediction sequence.
        """

        self.data_dir = data_dir
        self.stats_path = stats_path
        self.context_channels = context_channels
        self.optflow_channels = optflow_channels
        self.ts_channels = ts_channels
        self.ts_target_channels = ts_target_channels
        self.years = years[mode]
        self.stations = stations[mode]
        self.crop = crop
        self.mode = mode
        self.image_size = tuple(image_size) if image_size is not None else None
        self.use_target = use_target
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.n_samples = []
        self.year_mapping = {}

        # 打开时序数据的均值方差文件
        if self.stats_path is not None:
            with open(self.stats_path, "r") as fp:
                self.stats = json.load(fp)
        else:
            self.stats = None
        self.deeplake_ds = deeplake.load(self.data_dir, read_only=True)

        # 计算时序与云图的时间交集并将满足seq_len+pred_len长度的序列索引提取出来按顺序排序形成year_mapping
        for year in self.years:
            for station in self.stations:
                (
                    possible_starts_context,
                    possible_starts_station,
                ) = calculate_possible_starts(
                    self.deeplake_ds[f"{year}/context/time_utc"].numpy()[0],
                    self.deeplake_ds[f"{year}/{station}/time_utc"].numpy()[0],
                    frames_total=self.seq_len + self.pred_len,
                )
                n_in_group = len(possible_starts_context)
                self.n_samples.append(n_in_group)

                for i in range(self.n_samples[-1]):
                    step_id_station = possible_starts_station[i]
                    step_id_context = possible_starts_context[i]
                    self.year_mapping[i + sum(self.n_samples[:-1])] = (
                        str(year),
                        station,
                        step_id_station,
                        step_id_context,
                    )
        # Lazy loading
        self._coords = None
        self._elevation = None
        self._mean = {}
        self._std = {}
        self._mean_t = {}
        self._std_t = {}
        self._context_channel_ids = None
        self._optflow_channel_ids = None
        self._ts_channel_ids = {}
        self._ts_target_channel_ids = {}
        self._lat_slice = None
        self._lon_slice = None

    def get_stats(self, station):
        if station in self._mean:
            return self._mean[station], self._std[station]

        if self.stats is not None:
            mean = []
            std = []
            station_channels = (
                self.ts_channels
                if self.ts_channels is not None
                else self.stats[station]
            )
            for i, chan in enumerate(station_channels):
                mean.append(float(self.stats[station][chan]["mean"]))
                std.append(float(self.stats[station][chan]["std"]))
            mean = torch.tensor(mean).float().view(1, -1)
            std = torch.tensor(std).float().view(1, -1)
            self._mean[station] = mean
            self._std[station] = std
            return mean, std
        return None, None

    def get_stats_t(self, station):
        if station in self._mean_t:
            return self._mean_t[station], self._std_t[station]

        if self.stats is not None:
            mean = []
            std = []
            station_channels = (
                self.ts_target_channels
                if self.ts_target_channels is not None
                else self.stats[station]
            )
            for i, chan in enumerate(station_channels):
                mean.append(float(self.stats[station][chan]["mean"]))
                std.append(float(self.stats[station][chan]["std"]))
            mean = torch.tensor(mean).float().view(1, -1)
            std = torch.tensor(std).float().view(1, -1)
            self._mean_t[station] = mean
            self._std_t[station] = std
            return mean, std
        return None, None

    def get_coords(self, year):
        if self._coords is None:
            lat = self.deeplake_ds[f"{str(year)}/context/latitude"].numpy()

            lat = 2 * ((lat + 90) / 180) - 1
            lon = self.deeplake_ds[f"{str(year)}/context/longitude"].numpy()
            lon = 2 * ((lon + 180) / 360) - 1
            self._coords = np.stack(np.meshgrid(lat, lon), axis=0)
        return self._coords

    def get_elevation(self, year):
        if self._elevation is None:
            self._elevation = self.deeplake_ds[f"{str(year)}/context/elevation"].numpy()
        return self._elevation

    def get_channel_ids(
            self,
            context_tensor: deeplake.Tensor,
            optflow_tensor: deeplake.Tensor,
            timeseries_tensor: deeplake.Tensor,
            station: str
    ):
        """
        Get the list of channel indices to use for the context, optical flow and timeseries.
        Args:
            context_tensor (deeplake.Tensor): Context tensor to extract channel ids from.
            optflow_tensor (deeplake.Tensor): Optical flow tensor to extract channel ids from.
            timeseries_tensor (deeplake.Tensor): Timeseries tensor to extract channel ids from.
        """
        if self._context_channel_ids is None:
            if self.context_channels is not None:
                self._context_channel_ids = [
                    i
                    for i, k in enumerate(context_tensor.info["context_channels"])
                    for c in self.context_channels
                    if c == k
                ]
            else:
                self._context_channel_ids = [
                    i for i, k in enumerate(context_tensor.info["context_channels"])
                ]
            self._context_channel_ids = sorted(self._context_channel_ids)

        # if self._optflow_channel_ids is None:
        #     if self.optflow_channels is not None:
        #         self._optflow_channel_ids = [
        #             i
        #             for i, k in enumerate(optflow_tensor.info["optflow_channels"])
        #             for c in self.optflow_channels
        #             if c == k
        #         ]
        #     else:
        #         self._optflow_channel_ids = [
        #             i for i, k in enumerate(optflow_tensor.info["optflow_channels"])
        #         ]
        #     self._optflow_channel_ids = sorted(self._optflow_channel_ids)

        if station not in self._ts_channel_ids:
            if self.ts_channels is not None:
                self._ts_channel_ids[station] = [
                    i
                    for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
                    for c in self.ts_channels
                    if c == k
                ]
            else:
                self._ts_channel_ids[station] = [
                    i
                    for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
                ]
            self._ts_channel_ids[station] = sorted(self._ts_channel_ids[station])

        return (
            self._context_channel_ids,
            self._optflow_channel_ids,
            self._ts_channel_ids[station]
        )

    def get_target_channel_ids(self, ts_tensor, station: str):
        if station not in self._ts_target_channel_ids:
            if self.ts_target_channels is not None:
                self._ts_target_channel_ids[station] = [
                    i
                    for i, k in enumerate(ts_tensor.info["timeseries_channels"])
                    for c in self.ts_target_channels
                    if c == k
                ]
            else:
                self._ts_target_channel_ids[station] = [
                    i for i, k in enumerate(ts_tensor.info["timeseries_channels"])
                ]
            self._ts_target_channel_ids[station] = sorted(self._ts_target_channel_ids[station])
        return self._ts_target_channel_ids[station]

    def __len__(self) -> int:
        return sum(self.n_samples)

    def __getitem__(self, idx: int) -> dict:
        year, station, step_idx_station, step_idx_context = self.year_mapping[idx]
        # print(year)
        # print(step_idx_station)
        x_begin_index_ctx = step_idx_context
        x_end_index_ctx = x_begin_index_ctx + self.seq_len
        x_begin_index_ts = step_idx_station
        x_end_index_ts = x_begin_index_ts + self.seq_len
        # y_begin_index_ts = x_end_index_ts - self.label_len
        # y_end_index_ts = y_begin_index_ts + self.label_len + self.pred_len
        y_begin_index_ts = x_end_index_ts
        y_end_index_ts = y_begin_index_ts + self.pred_len

        # time_u = self.deeplake_ds[f"{year}/{station}/time_utc"]
        if self.use_target:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                0,
                x_begin_index_ts:x_end_index_ts,
                ].numpy()
            )
        else:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                0,
                x_begin_index_ts:x_end_index_ts + self.seq_len,
                ].numpy()
            )

        mean, std = self.get_stats(station)

        ts_tensor = self.deeplake_ds[f"{year}/{station}/data"]
        context_tensor = self.deeplake_ds[f"{year}/context/data"]
        optflow_tensor = self.deeplake_ds[f"{year}/ctx_opt_flow/data"]
        context_channel_ids, optflow_channel_ids, ts_channel_ids = self.get_channel_ids(
            context_tensor, optflow_tensor, ts_tensor, station
        )
        # print(station)
        # print(x_begin_index_ts)
        # print(x_end_index_ts)
        # print(ts_channel_ids)
        target_channel_ids = self.get_target_channel_ids(ts_tensor, station)
        # print(self._ts_channel_ids)
        # print(ts_channel_ids)
        # print(type(ts_channel_ids))
        timseries_data = torch.from_numpy(
            ts_tensor[
                0,
                x_begin_index_ts:x_end_index_ts,
                ts_channel_ids,
            ].numpy()
        )
        if mean is not None:
            timseries_data = (timseries_data - mean) / std
            # timseries_data = (timseries_data - std) / (mean - std)

        target = torch.from_numpy(
            ts_tensor[
                0,
                y_begin_index_ts:y_end_index_ts,
                target_channel_ids,
            ].numpy()
        )
        target_previous = torch.from_numpy(
            ts_tensor[
                0,
                x_begin_index_ts:x_end_index_ts,
                target_channel_ids
            ].numpy()
        )
        # mean, std = self.get_stats_t(station)
        # if mean is not None:
        #     target = (target - mean) / std
        #     target_previous = (target_previous - mean) / std

        context_data = torch.from_numpy(
            context_tensor[
                x_begin_index_ctx:x_end_index_ctx,
                context_channel_ids,
            ].numpy()
        )
        # optflow_data = torch.from_numpy(
        #     optflow_tensor[
        #         x_begin_index_ctx:x_end_index_ctx,
        #         optflow_channel_ids,
        #     ].numpy()
        # )

        context_coords = torch.from_numpy(self.get_coords(year))
        context_elevation = torch.from_numpy(self.get_elevation(year))

        station_elevation = (
            torch.Tensor(
                [
                    ts_tensor.info["elevation"],
                ]
            ) - context_elevation.mean()) / context_elevation.std()
        # station_elevation = station_elevation.unsqueeze(0)
        station_coords = [
            2 * (ts_tensor.info["coordinates"][0] + 90) / 180 - 1,
            2 * (ts_tensor.info["coordinates"][1] + 180) / 360 - 1,
        ]
        station_coords = torch.Tensor(station_coords)[(...,) + (None,) * 2]
        context_elevation = (context_elevation - context_elevation.mean()) / context_elevation.std()

        if self.image_size is not None:
            # optflow_data = F.interpolate(
            #     optflow_data, size=self.image_size, mode="bilinear", align_corners=True
            # )
            context_data = F.interpolate(
                context_data, size=self.image_size, mode="bilinear", align_corners=True
            )
            context_coords = F.interpolate(
                context_coords.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            context_elevation = F.interpolate(
                context_elevation.unsqueeze(0).unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze()

        context_elevation = context_elevation.expand(context_data.shape[0], 1, -1, -1)
        station_elevation = station_elevation.expand(timseries_data.shape[0], -1)

        H, W = context_data.shape[-2:]

        dates_df = pd.DataFrame({"date": time_utc})
        time = time_features(dates_df, True, "t")
        time = torch.from_numpy(time)[(...,) + (None,) * 2].repeat(1, 1, H, W)
        # months = torch.from_numpy(time_utc.dt.month.values)[(...,) + (None,) * 3].repeat(
        #     1, 1, H, W
        # )
        # days = torch.from_numpy(time_utc.dt.day.values)[(...,) + (None,) * 3].repeat(
        #     1, 1, H, W
        # )
        # hours = torch.from_numpy(time_utc.dt.hour.values)[(...,) + (None,) * 3].repeat(
        #     1, 1, H, W
        # )
        # minutes = torch.from_numpy(time_utc.dt.minute.values)[
        #     (...,) + (None,) * 3
        #     ].repeat(1, 1, H, W)
        #
        # time = torch.cat([months, days, hours, minutes], dim=1)

        return_tensors = {
            "context": context_data,
            "optical_flow": [],
            "context_coordinates": context_coords,
            "context_elevation": context_elevation,

            "timeseries": timseries_data,
            "time_embedding": time,
            "station_coordinates": station_coords,
            "station_elevation": station_elevation,
            "ts_mean": mean,
            "ts_std": std
        }
        if self.use_target:
            return_tensors["target"] = target
            return_tensors["target_previous"] = target_previous
        return return_tensors


if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    os.environ['PYTHONHASHSEED'] = str(1)
    torch.manual_seed(1)
    # torch_npu.npu.manual_seed(seed+rank)
    # torch_npu.npu.manual_seed_all(seed+rank)
    cudnn.benchmark = False
    cudnn.deterministic = True
    data_dir= '/home/zhangfeng/tangjh/oth/dataset'
    # data_dir='/home/zhangfeng/tangjh/oth/dataset'
    # stats_path='/home/zhangfeng/tangjh/oth/Forecast_pretrain/stats_multi1.json'
    stats_path = '/home/zhangfeng/tangjh/oth/Forecast_pretrain/stats_multi2.json'
    context_channels=[ 'IR_108', 'VIS006', 'VIS008', 'WV_062', 'WV_073' ]
    optflow_channels=['IR_108_vx', 'IR_108_vy', 'WV_062_vx', 'WV_062_vy', 'WV_073_vx', 'WV_073_vy']
    ts_channels=['GHI', 'PoPoPoPo [hPa]', 'DIF [W/m**2]', 'DIR [W/m**2]',  'ghi', 'dni', 'dhi']
    ts_target_channels=['GHI']
    years={'train':[ "2008_nonhrv", "2009_nonhrv","2010_nonhrv", "2011_nonhrv","2012_nonhrv", "2013_nonhrv", "2014_nonhrv", "2015_nonhrv", "2016_nonhrv" ],
           'val':[ "2017_nonhrv", "2018_nonhrv", "2019_nonhrv" ],
           'test':[ "2020_nonhrv", "2021_nonhrv", "2022_nonhrv" ]}
    stations={'train':[ "PCCI_20082022_IZA", "PCCI_20082022_CNR", "PCCI_20082022_PAL" , "PCCI_20082022_PAY", "PCCI_20082022_CAB"],
              'val':[ "PCCI_20082022_PAY" ],
              'test':[ "PCCI_20082022_CAB" ]}
    image_size=None
    crop=None
    seq_len=48
    label_len=0
    pred_len=48
    use_target=True

    mydata = TSContextDataset(data_dir,stats_path,context_channels,optflow_channels,ts_channels,ts_target_channels, years, stations, 'test', use_target, image_size, crop, seq_len, label_len, pred_len)
    numbers = list(range(1, 2000))
    random.shuffle(numbers)
    indices=numbers[:1000]
    print(indices)
    for i in indices:
        data = mydata[i]
        print(f"{i}:{data['timeseries'].shape}")

    # def get_channel_ids(
    #         context_tensor: deeplake.Tensor,
    #         timeseries_tensor: deeplake.Tensor,
    #         station: str
    # ):
    #     _context_channel_ids = None
    #     _optflow_channel_ids = None
    #     _ts_channel_ids = {}
    #     context_channels = ["IR_039", "IR_087", "IR_108", "VIS006", "VIS008", "WV_062", "WV_073"]
    #     ts_channels = ['GHI', 'DIF [W/m**2]', 'DIR [W/m**2]', 'PoPoPoPo [hPa]', 'dhi', 'dni', 'ghi']
    #     if _context_channel_ids is None:
    #         if context_channels is not None:
    #             _context_channel_ids = [
    #                 i
    #                 for i, k in enumerate(context_tensor.info["context_channels"])
    #                 for c in context_channels
    #                 if c == k
    #             ]
    #         else:
    #             _context_channel_ids = [
    #                 i for i, k in enumerate(context_tensor.info["context_channels"])
    #             ]
    #         _context_channel_ids = sorted(_context_channel_ids)
    #
    #     if station not in _ts_channel_ids:
    #         if ts_channels is not None:
    #             _ts_channel_ids[station] = [
    #                 i
    #                 for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
    #                 for c in ts_channels
    #                 if c == k
    #             ]
    #         else:
    #             _ts_channel_ids[station] = [
    #                 i
    #                 for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
    #             ]
    #         _ts_channel_ids[station] = sorted(_ts_channel_ids[station])
    #
    #     return (
    #         _context_channel_ids,
    #         _ts_channel_ids[station]
    #     )
    # deeplake_ds = deeplake.load('/home/ma-user/work/dataset', read_only=True)
    # ts_tensor = deeplake_ds[f"2008_nonhrv/PCCI_20082022_CAB/data"]
    # context_tensor = deeplake_ds[f"2008_nonhrv/context/data"]
    # optflow_tensor = deeplake_ds[f"2008_nonhrv/ctx_opt_flow/data"]
    # context_channel_ids, ts_channel_ids = get_channel_ids(
    #     context_tensor, ts_tensor, "PCCI_20082022_CAB"
    # )
    # print(ts_channel_ids)

