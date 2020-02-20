import os
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.retinanet import RetinaNet as BaseRetinaNet

from pyronan.model import Model
from pyronan.models.utils import draw, draw_batch, is_backbone_grad


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).parent / "retinanet.yaml"))
    cfg.merge_from_list(args)
    cfg.freeze()
    return cfg


class RetinaNet(Model):
    def __init__(self, args, num_classes):
        self.cfg_dict = {
            "MODEL.RETINANET.NUM_CLASSES": num_classes,
            "MODEL.DEVICE": "cuda" if args.gpu else "cpu",
            "INPUT.MIN_SIZE_TRAIN": (args.min_size,),
            "INPUT.MAX_SIZE_TRAIN": args.max_size,
        }
        cfg_list = sum([[k, v] for k, v in self.cfg_dict.items()], [])
        cfg = setup_cfg(cfg_list)
        nn_module = BaseRetinaNet(cfg)
        self.nms_iou = getattr(args, "nms_iou", None)
        self.min_size = args.min_size
        self.max_size = args.max_size
        super().__init__(nn_module, args)

    def step(self, batch, set_):
        self.batch = batch
        loss_dict = self.nn_module(self.batch)
        loss = sum(loss for loss in loss_dict.values())
        if set_ == "train":
            self.update(loss)
        loss_dict["loss"] = loss
        return {k: v.item() for k, v in loss_dict.items()}

    def get_image(self, cutoff=0):
        self.nn_module.eval()
        instances_list = self.nn_module(self.batch)
        predictions = []
        for instances in instances_list:
            instances = instances["instances"].to("cpu")
            predictions.append(
                {
                    "labels": instances.pred_classes.numpy(),
                    "scores": instances.scores.detach().numpy(),
                    "boxes": instances.pred_boxes.numpy(),
                }
            )
        self.nn_module.train()
        return draw_batch(self.batch, predictions, cutoff)
