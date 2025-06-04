from importlib.util import find_spec
from typing import Callable
import torch.distributed as dist
from omegaconf import DictConfig

def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.
    Makes multirun more resistant to failure.
    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig, logger, rank):

        # apply extra utilities
        # execute the task
        try:
            task_func(cfg=cfg, logger=logger, rank=rank)
        except Exception as ex:
            dist.destroy_process_group()
            # log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            # path = Path(cfg.paths.output_dir, "exec_time.log")
            # content = (
            #     f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            # )
            # save_file(
            #     path, content
            # )  # save task execution time (even if exception occurs)
            close_loggers(logger)  # close loggers (even if exception occurs so multirun won't fail)

        # log.info(f"Output dir: {cfg.paths.output_dir}")
    return wrap

def close_loggers(logger) -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    logger.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb
        if wandb.run:
            logger.info("Closing wandb!")
            wandb.finish()
