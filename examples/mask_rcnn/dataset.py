#!/usr/bin/env python3
import torch
from torchvision.transforms import functional as F

from Positional_Reconstruction_Network.src.datasets.dataset_ue import Dataset_ue


def compute_box(binmask):
    xx, yy = binmask.nonzero()
    return xx.min(), yy.min(), xx.max(), yy.max()


class Dataset_custom(Dataset_ue):
    def __init__(self, args, set_):
        super().__init__(args, set_)

    def getitem(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        raw_mask, idx_list = self._get_mask_and_status(dir_, frame_idx)
        boxes, labels, area, iscrowd, masks = [], [], [], [], []
        for label, idx in enumerate(idx_list[1:], 1):
            for i in idx:
                xx, yy = (raw_mask == i).nonzero()
                if len(xx) == 0 or len(yy) == 0:
                    continue
                bb = [yy.min(), xx.min(), yy.max(), xx.max()]
                if bb[0] == bb[2] or bb[1] == bb[3]:
                    continue
                boxes.append(bb)
                labels.append(label)
                area.append((bb[2] - bb[0]) * (bb[3] - bb[1]))
                iscrowd.append(0)
                masks.append(raw_mask == i)
        if len(boxes) == 0:
            boxes.append([0, 0, self.hw, self.hw])
            labels.append(5)
            area.append(self.hw ** 2)
            iscrowd.append(0)
            masks.append(raw_mask * 0)
        return {
            "boxes": torch.FloatTensor(boxes),
            "labels": torch.LongTensor(labels),
            "area": torch.LongTensor(area),
            "iscrowd": torch.ByteTensor(iscrowd),
            "masks": torch.ByteTensor(masks),
        }

    def get_height_and_width(self, _):
        return self.hw, self.hw

    def __getitem__(self, index):
        video_idx = index // (self.n_frames * self.step)
        frame_idx = index % (self.n_frames * self.step)
        image = self.get_image(video_idx, frame_idx)
        target = self.getitem(video_idx, frame_idx)
        image = F.to_tensor(image)
        target["image_id"] = torch.LongTensor([index])
        return image, target
