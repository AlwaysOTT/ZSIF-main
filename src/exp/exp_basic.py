import torch


class Exp_Basic(object):
    def __init__(self):
        self.logger = None
        # self.gpu = None
        # self.device = None

    def acquire_device(self, gpu, cfg):
        pass

    def set_logger(self, logger):
        self.logger = logger

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def val(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
