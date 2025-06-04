import os
import pyrootutils
import torch
from hydra.utils import instantiate
from typing import Optional
import hydra
from omegaconf import DictConfig
import torch.multiprocessing as mp
import torch.distributed as dist

from src.utils import utils
from src.utils.log import setup_logger
from src.utils.tools import random_seed
# from src.utils.torch_dist import synchronize

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["main.py"],
    pythonpath=True,
    dotenv=True,
)
# log = utils.get_pylogger(__name__)

@hydra.main(version_base="1.2", config_path=str(root) + "/configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.devices
    nr_gpu = len(cfg.devices.split(","))
    nr_machine = int(os.getenv("MACHINE_TOTAL", "1"))
    cfg.world_size = nr_gpu * nr_machine
    if cfg.world_size > 1:
        processes = []
        for rank in range(nr_gpu):
            p = mp.Process(target=main_worker, args=(rank, nr_gpu, cfg))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
    else:
        main_worker(0, nr_gpu, cfg)
    dist.destroy_process_group()

def main_worker(gpu, nr_gpu, cfg):
    rank = gpu
    if cfg.world_size > 1:
        rank += int(os.getenv("MACHINE_RANK", "0")) * nr_gpu
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=rank,
        )
        # synchronize()

    if cfg.seed is not None:
        print("seed now ！！")
        random_seed(cfg.seed, rank)

    file_name = os.path.join("./ts_context_logs/outputs", cfg.exp_main.name)
    file_name = os.path.join(file_name, cfg.task_name)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)
    logger = setup_logger(file_name, distributed_rank=rank, filename="log.txt", mode="a")
    if cfg.train:
        build_train(cfg, logger, rank)
    if cfg.test:
        build_test(cfg, logger, rank)
    torch.cuda.empty_cache()

@utils.task_wrapper
def build_train(cfg, logger, rank):
    exp = instantiate(cfg.exp_main)
    exp.set_logger(logger)
    exp.acquire_device(rank, cfg)
    exp.train()
    # exp.test()

@utils.task_wrapper
def build_test(cfg, logger, rank):
    exp = instantiate(cfg.exp_main)
    exp.set_logger(logger)
    exp.acquire_device(rank, cfg)
    exp.test()


# def fix_DictConfig(cfg: DictConfig):
#     """fix all vars in the cfg config
#     this is an in-place operation"""
#     keys = list(cfg.keys())
#     for k in keys:
#         print(type(cfg[k]))
#         if type(cfg[k]) is DictConfig:
#             fix_DictConfig(cfg[k])
#         else:
#             setattr(cfg, k, getattr(cfg, k))


if __name__ == "__main__":
    main()
    # mp.spawn(main, nprocs=2)
