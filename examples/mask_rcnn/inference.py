from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from dataset import Dataset_custom
from examples.mask_rcnn.model import MaskRCNN
from pyronan.model import make_model, parser_model
from pyronan.utils.image import ti, tis
from pyronan.utils.misc import parse_slice
from sandbox.utils import plot_position


def parse_args(argv=None):
    parser = ArgumentParser(parents=[parser_model])
    parser.add_argument(
        "--input_prefix", type=Path, default="/sata/rriochet/intphys2019/traintest"
    )
    parser.add_argument("--hw", type=int, default=288)
    parser.add_argument("--N_o", type=int, default=6, help="number of balls")
    parser.add_argument("--video_slice", type=parse_slice, default=slice(None))
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--bsz", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    return args


def process(pred, cutoff, hw):
    scores = pred["scores"].detach().numpy()
    boxes = pred["boxes"].detach().numpy()
    boxes = boxes[scores > cutoff]
    boxes = boxes * 2 / hw - 1
    boxes[:, 1] *= -1
    boxes[:, 3] *= -1
    return boxes


def visualize(image, boxes):
    image = ti(image)
    image = plot_position(image, boxes, box=True, scale=1)
    tis(image)


def main():
    args = parse_args()
    print(args)
    dataset = Dataset_custom(args, "val")
    loader = DataLoader(dataset, 1, shuffle=True)
    model = make_model(MaskRCNN, args, args.gpu, args.data_parallel, args.load)
    model.nn_module.eval()
    for batch in loader:
        pred = model(batch[0])
        boxes = process(pred[0], args.cutoff, batch[0].shape[-1])
        visualize(batch[0].numpy(), boxes)


if __name__ == "__main__":
    main()
