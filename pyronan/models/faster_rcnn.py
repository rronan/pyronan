import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image
from pyronan.model import Model
from pyronan.utils.draw import draw_detection_batched
from pyronan.utils.image import COLOR_LIST
from pyronan.utils.torchutil import is_backbone_grad
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class RCNN(Model):
    def __init__(self, nn_module, args):
        self.nms_iou = getattr(args, "nms_iou", None)
        self.cutoff = getattr(args, "cutoff", 0)
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

    @staticmethod
    def draw(images, detections, cutoff):
        image_batch, detection_batch, colors_batch = [], [], []
        for image, detection in zip(images, detections):
            labels = detection["labels"].detach().cpu().numpy()
            c = labels != 0
            if "scores" in detection:
                c *= detection["scores"].detach().cpu().numpy() > cutoff
            detection_batch.append(
                {"boxes": detection["boxes"].detach().cpu().numpy()[c]}
            )
            colors_batch.append([COLOR_LIST[i] for i in labels[c]])
            image = (image * 255).numpy().astype(np.uint8).transpose((1, 2, 0))
            image_batch.append(Image.fromarray(image))
        res = draw_detection_batched(image_batch, detection_batch, colors_batch)
        return res

    def get_image(self):
        self.nn_module.eval()
        predictions = self.nn_module([x.to(self.device) for x in self.batch[0]])
        self.nn_module.train()
        images, targets = self.batch
        im_list = [self.draw(images, y, self.cutoff) for y in [targets, predictions]]
        res = np.concatenate(im_list, axis=0)[np.newaxis]
        return res.astype("uint8").transpose((0, 3, 1, 2))

    def __call__(self, x):
        y = self.nn_module(x)
        if self.nms_iou is not None:
            keep = torchvision.ops.nms(y["boxes"], y["scores"], self.nms_iou)
            for k in ["boxes", "labels", "masks"]:
                y[k] = y[k].select(keep)
        return y


def fasterrcnn_resnet_fpn(name, pretrained_backbone=True, **kwargs):
    backbone = resnet_fpn_backbone(name, pretrained_backbone)
    model = faster_rcnn.FasterRCNN(backbone, 1, **kwargs)
    return model


class FasterRCNN(RCNN):
    def __init__(self, args, num_classes=None):
        self.num_classes = num_classes if num_classes is not None else args.num_classes
        self.resnet_name = getattr(args, "resnet_name", "resnet50")
        nn_module = fasterrcnn_resnet_fpn(
            name=self.resnet_name, min_size=args.min_size, max_size=args.max_size,
        )
        in_features = nn_module.roi_heads.box_predictor.cls_score.in_features
        nn_module.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features, self.num_classes
        )
        super().__init__(nn_module, args)


class MaskRCNN(RCNN):
    def __init__(self, args, num_classes=None):
        self.num_classes = num_classes if num_classes is not None else args.num_classes
        nn_module = faster_rcnn.maskrcnn_resnet50_fpn(
            pretrained=args.pretrained, min_size=args.min_size, max_size=args.max_size
        )
        in_features = nn_module.roi_heads.box_predictor.cls_score.in_features
        nn_module.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features, self.num_classes
        )
        in_features_mask = nn_module.roi_heads.mask_predictor.conv5_mask.in_channels
        nh = 256
        nn_module.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, nh, self.num_classes
        )
        super().__init__(nn_module, args)
