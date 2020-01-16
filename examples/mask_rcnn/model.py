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

# torch.multiprocessing.set_sharing_strategy("file_system")


def is_backbone_grad(lr):
    for kv in lr:
        if type(kv) is float:
            return True
        else:
            if kv[0] == "backbone":
                return True
    return False


class MaskRCNN(Model):
    def __init__(self, args):
        self.device = "cpu"
        self.is_data_parallel = False
        # load an instance segmentation model pre-trained pre-trained on COCO
        nn_module = maskrcnn_resnet50_fpn(pretrained=args.pretrained)

        # get number of input features for the classifier
        in_features = nn_module.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        nn_module.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, args.num_classes
        )

        # now get the number of input features for the mask classifier
        in_features_mask = nn_module.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        nn_module.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, args.num_classes
        )
        backbone_grad = is_backbone_grad(args.lr)
        print("training backbone", backbone_grad)
        nn_module.backbone.requires_grad_(backbone_grad)
        super().__init__(nn_module, args)

    def data_parallel(self):
        self.nn_module.backbone = nn.DataParallel(self.nn_module.backbone)
        # self.nn_module.rpn = nn.DataParallel(self.nn_module.rpn)
        # self.nn_module.roi_heads = nn.DataParallel(self.nn_module.roi_heads)
        self.is_data_parallel = True

    def step(self, batch, set_):
        images, targets = [], []
        for image, target in zip(*batch):
            if len(target["boxes"]) == 0:
                continue
            images.append(image.to(self.device))
            targets.append({k: v.to(self.device) for k, v in target.items()})
        if len(images) == 0:
            return {"loss": 0}
        if set_ == "train":
            loss_dict = self.nn_module(images, targets)
        else:
            with torch.no_grad():
                loss_dict = self.nn_module(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        if set_ == "train":
            self.update(loss)
        loss_dict["loss"] = loss
        self.images = batch[0]
        self.pred = targets
        self.true_boxes = [x["boxes"].numpy().copy().tolist() for x in batch[1]]
        self.true_labels = [x["labels"].numpy().copy().tolist() for x in batch[1]]
        return {k: v.item() for k, v in loss_dict.items()}

    def get_image(self):
        res = []
        for image_tensor, boxes, labels in zip(
            self.images, self.true_boxes, self.true_labels
        ):
            image_array = image_tensor.numpy().transpose((1, 2, 0))
            im = ti(image_array)
            draw = ImageDraw.Draw(im)
            for box, label in zip(boxes, labels):
                draw.rectangle(box, outline=tuple(COLOR_LIST[label]))
            res.append(np.array(im))
        return np.array(res).transpose((0, 3, 1, 2))
