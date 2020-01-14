import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pyronan.model import Model


class MaskRCNN(Model):
    def __init__(self, args):
        self.device = "cpu"
        self.is_data_parallel = False
        # load an instance segmentation model pre-trained pre-trained on COCO
        nn_module = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=args.pretrained
        )

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
        if not "backbone" in [x[0] for x in args.lr_list]:
            nn_module.backbone.requires_grad_(False)
        super().__init__(nn_module, args)

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
        return {k: v.item() for k, v in loss_dict.items()}
