import numpy as np
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pyronan.model import Model
from pyronan.utils.draw import draw_detection_batch
from pyronan.utils.torchutil import is_backbone_grad


class RCNN(Model):
    def __init__(self, nn_module, args):
        self.nms_iou = getattr(args, "nms_iou", None)
        self.min_size = args.min_size
        self.max_size = args.max_size
        backbone_grad = is_backbone_grad(args.lr)
        print("training backbone", backbone_grad)
        nn_module.backbone.requires_grad_(backbone_grad)
        super().__init__(nn_module, args)

    def data_parallel(self):
        self.nn_module.backbone = nn.DataParallel(self.nn_module.backbone)
        self.is_data_parallel = True

    def step(self, batch, set_):
        self.batch = batch
        images, targets = [], []
        for image, target in zip(*batch):
            images.append(image.to(self.device))
            targets.append({k: v.to(self.device) for k, v in target.items()})
        loss_dict = self.nn_module(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        if set_ == "train":
            self.update(loss)
        loss_dict["loss"] = loss
        return {k: v.item() for k, v in loss_dict.items()}

    def get_input_to_model(self):
        return [image.to(self.device) for image in self.batch[0]]

    def get_image(self, cutoff=0):
        self.nn_module.eval()
        predictions = self.nn_module([x.to(self.device) for x in self.batch[0]])
        self.nn_module.train()
        return draw_detection_batch(*self.batch, predictions, cutoff)

    def __call__(self, x):
        y = self.nn_module(x)
        if self.nms_iou is not None:
            keep = torchvision.ops.nms(y["boxes"], y["scores"], self.nms_iou)
            for k in ["boxes", "labels", "masks"]:
                y[k] = y[k].select(keep)
        return y


class FasterRCNN(RCNN):
    def __init__(self, args, num_classes=None):
        self.num_classes = num_classes if num_classes is not None else args.num_classes
        nn_module = fasterrcnn_resnet50_fpn(
            pretrained=args.pretrained, min_size=args.min_size, max_size=args.max_size
        )
        in_features = nn_module.roi_heads.box_predictor.cls_score.in_features
        nn_module.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        super().__init__(nn_module, args)


class MaskRCNN(RCNN):
    def __init__(self, args, num_classes=None):
        self.num_classes = num_classes if num_classes is not None else args.num_classes
        nn_module = maskrcnn_resnet50_fpn(
            pretrained=args.pretrained, min_size=args.min_size, max_size=args.max_size
        )
        in_features = nn_module.roi_heads.box_predictor.cls_score.in_features
        nn_module.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        in_features_mask = nn_module.roi_heads.mask_predictor.conv5_mask.in_channels
        nh = 256
        nn_module.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, nh, self.num_classes
        )
        super().__init__(nn_module, args)
