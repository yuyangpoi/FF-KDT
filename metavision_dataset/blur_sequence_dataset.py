import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from metavision_dataset.my_gpu_esim_v3 import My_GPUEBSimCorners


base_params = {
    'dataset_path': '',
    'num_workers': 0,
    'batch_size': 2,
    'num_tbins': 7,     # BPTT
    'warping_event_volume_depth': 10,
    'exposure_event_volume_depth': 10,
    'height': 240,
    'width': 320,
    'min_frames_per_video': 200,
    'max_frames_per_video': 6000,
    'raw_number_of_heatmaps': 41,
    'randomize_noises': True,
    'data_device': 'cuda:0',

    'needed_number_of_heatmaps': 5,
    'min_blur_ratio': 0.2,
    'max_blur_ratio': 0.7,
    'random_gamma_transform': True,
    'random_pepper_noise': True,
    'seed': 6,

    'epochs': 100,

    'max_optical_flow_threshold_random_list': (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
}


class Blur_sequnece_dataloader(pl.LightningDataModule):
    """
    Simulation gives events + frames + corners
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.split_names = ['train', 'val']


    def get_dataloader(self, dataset_path, current_epoch):
        current_seed = self.hparams.seed + current_epoch
        print('current_seed: ', current_seed)


        dataloader = My_GPUEBSimCorners.from_params(
            dataset_path,
            self.hparams.num_workers,
            self.hparams.batch_size,
            self.hparams.num_tbins,
            self.hparams.warping_event_volume_depth,
            self.hparams.exposure_event_volume_depth,
            self.hparams.height,
            self.hparams.width,
            self.hparams.min_frames_per_video,
            self.hparams.max_frames_per_video,
            self.hparams.raw_number_of_heatmaps,  # current_raw_number_of_heatmaps,
            self.hparams.randomize_noises,
            self.hparams.data_device,

            self.hparams.needed_number_of_heatmaps,
            self.hparams.min_blur_ratio,
            self.hparams.max_blur_ratio,
            self.hparams.random_gamma_transform,
            self.hparams.random_pepper_noise,
            current_seed,

            max_optical_flow_threshold_random_list=self.hparams.max_optical_flow_threshold_random_list
        )
        return dataloader

    def train_dataloader(self):
        path = os.path.join(self.hparams.dataset_path, self.split_names[0])
        if self.trainer is not None:
            return self.get_dataloader(path, self.trainer.current_epoch)
        else:
            return self.get_dataloader(path, 0)

    def val_dataloader(self):
        path = os.path.join(self.hparams.dataset_path, self.split_names[1])
        if self.trainer is not None:
            return self.get_dataloader(path, self.trainer.current_epoch)
        else:
            return self.get_dataloader(path, 0)





