import numpy as np
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import relu_fn
from segmentation_models_pytorch.unet.decoder import UnetDecoder

from pyronan.model import Model
from pyronan.utils.image import ti

IDX_DICT = {3: [1, 4, 7, 17], 5: [2, 7, 12, 26]}
ENCODER_CHANNELS = {3: (1536, 136, 48, 32, 24), 5: (2048, 176, 64, 40, 24)}


class EfficientNet_Encoder(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.idx_list = IDX_DICT[b]
        self.encoder_channels = ENCODER_CHANNELS[b]
        self.model = EfficientNet.from_pretrained(f"efficientnet-b{b}")

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))
        global_features = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.idx_list:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()
        return global_features


class Core_EfficientNet_unet(nn.Module):
    def __init__(self, b, num_channels):
        super().__init__()
        self.model_encoder = EfficientNet_Encoder(b)
        self.model_decoder = nn.Sequential(
            UnetDecoder(
                encoder_channels=self.model_encoder.encoder_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                final_channels=num_channels,
                use_batchnorm=True,
                center=False,
            ),
            nn.LogSoftmax(dim=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True)
        )
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):
        global_features = self.model_encoder(x)
        seg_feature = self.model_decoder(global_features)
        return seg_feature


class EfficientNet_unet(Model):
    def __init__(self, args):
        self.hw = args.hw
        nn_module = Core_EfficientNet_unet(args.b, args.nc_out)
        super().__init__(nn_module, args)
        self.loss = nn.NLLLoss()

    def save_im(self, path):
        mask_model = ti(np.exp(self.pred.cpu().numpy()))
        mask_model.save(f"{path}_mask_model.png")
        mask_true = ti(self.y.cpu().numpy()[:, np.newaxis])
        mask_true.save(f"{path}_mask_true.png")
