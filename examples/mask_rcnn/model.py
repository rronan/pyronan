import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pyronan.model import Model


class MaskRCNN(Model):
    def __init__(self, args):
        # load an instance segmentation model pre-trained pre-trained on COCO
        nn_module = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

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
        super().__init__(nn_module, args)

    def step_train(self, images, targets):
        self.nn_module.train()
        loss_dict = self.nn_module(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.update(loss)
        loss_dict["loss"] = loss
        return {k: v.item() for k, v in loss_dict.items()}

    def step_eval(self, images, targets):
        # self.nn_module.eval()
        loss_dict = self.nn_module(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_dict["loss"] = loss
        return {k: v.item() for k, v in loss_dict.items()}

    def step(self, batch, set_):
        images = list(image.to(self.device) for image in batch[0])
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch[1]]
        if set_ == "train":
            return self.step_train(images, targets)
        elif set_ == "val":
            with torch.no_grad():
                return self.step_eval(images, targets)
