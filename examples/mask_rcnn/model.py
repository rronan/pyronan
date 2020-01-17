import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pyronan.model import Model
from pyronan.utils.image import COLOR_LIST, ti, tis

torch.multiprocessing.set_sharing_strategy("file_system")


def is_backbone_grad(lr):
    for kv in lr:
        if type(kv) is float:
            return True
        else:
            if kv[0] == "backbone":
                return True
    return False


def draw(args):
    image_array, boxes, labels = args
    im = ti(image_array)
    draw = ImageDraw.Draw(im)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline=tuple(COLOR_LIST[label]))
    return np.array(im)


class MaskRCNN(Model):
    def __init__(self, args):
        self.device = "cpu"
        self.is_data_parallel = False
        nc = args.num_classes
        nn_module = maskrcnn_resnet50_fpn(pretrained=args.pretrained)
        in_features = nn_module.roi_heads.box_predictor.cls_score.in_features
        nn_module.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc)
        in_features_mask = nn_module.roi_heads.mask_predictor.conv5_mask.in_channels
        nh = 256
        nn_module.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, nh, nc)
        backbone_grad = is_backbone_grad(args.lr)
        print("training backbone", backbone_grad)
        nn_module.backbone.requires_grad_(backbone_grad)
        super().__init__(nn_module, args)

    def data_parallel(self):
        self.nn_module.backbone = nn.DataParallel(self.nn_module.backbone)
        self.is_data_parallel = True

    def step(self, batch, set_):
        self.images, self.true_boxes, self.true_labels = [], [], []
        images, targets = [], []
        for image, target in zip(*batch):
            if len(target["boxes"]) == 0:
                continue
            self.images.append(image.numpy().transpose((1, 2, 0)))
            self.true_boxes.append(target["boxes"].numpy().copy().tolist())
            self.true_labels.append(target["labels"].numpy().copy().tolist())
            images.append(image.to(self.device))
            targets.append({k: v.to(self.device) for k, v in target.items()})
        if len(images) == 0:
            return {"loss": 0}
        loss_dict = self.nn_module(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        if set_ == "train":
            self.update(loss)
        loss_dict["loss"] = loss
        self.images = [x.detach().cpu().numpy().transpose((1, 2, 0)) for x in images]
        self.preds = targets
        return {k: v.item() for k, v in loss_dict.items()}

    def get_image(self):
        true = list(map(draw, zip(self.images, self.true_boxes, self.true_labels)))
        pred_boxes = [x["boxes"].cpu().numpy() for x in self.preds]
        pred_labels = [x["labels"].cpu().numpy() for x in self.preds]
        pred = list(map(draw, zip(self.images, pred_boxes, pred_labels)))
        res = np.concatenate([true, pred], axis=1).transpose((0, 3, 1, 2))
        return (res * 255).astype("uint8")
