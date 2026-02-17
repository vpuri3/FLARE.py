#
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

import mlutils

__all__ = [
    'Callback',
]

#======================================================================#
class Callback(mlutils.Callback):
    def __init__(self, case_dir: str, metadata: dict):
        super().__init__(case_dir)
        self.metadata = metadata

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):

        return

#======================================================================#
#