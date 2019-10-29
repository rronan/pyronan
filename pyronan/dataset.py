import re

import numpy as np
import torch
import torch.utils.data
from torch._six import container_abcs, int_classes, string_classes


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, set_):
        self.set_ = set_
        self.count = args.count

    def make_data(self, input_prefix, set_):
        self.data = ""

    def count_data(self, c=-1):
        self.n_videos, self.n_frames, *_ = self.data.shape
        self.count = self.n_videos * self.n_frames
        self.count = self.count if c < 0 else min(self.count, c)

    def print_stats(self):
        d = {"input": self.data}
        print("*****", self.set_, sep="\n")
        for key, value in d.items():
            print(f"{key} max :", np.max(value))
            print(f"{key} min :", np.min(value))
            print(f"{key} mean :", np.mean(value))
            print(f"{key} std :", np.std(value))
            print(f"{key} shape :", value.shape)
        print(f"n samples {self.set_}: {self.count}")

    def get_input(self, video_idx, frame_idx):
        pass

    def get_target(self, video_idx, frame_idx):
        pass

    def getitem(self, video_idx, frame_idx):
        input_ = self.get_input(video_idx, frame_idx)
        target = self.get_target(video_idx, frame_idx)
        return input_, target

    def __getitem__(self, index):
        video_idx = index // self.n_frames
        frame_idx = index % self.n_frames
        return self.getitem(video_idx, frame_idx)

    def __len__(self):
        return self.count
