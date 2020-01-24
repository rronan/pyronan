from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from pyronan.examples.mask_rcnn.model import MaskRCNN
from pyronan.examples.mask_rcnn.train import make_loader
from pyronan.model import make_model, parser_model
from pyronan.train import parser_train
from pyronan.utils.image import COLOR_LIST, ti, tis
from pyronan.utils.misc import parse_slice


def parse_args(argv=None):
    parser = ArgumentParser(parents=[parser_model, parser_train])
    parser.add_argument(
        "--input_prefix", type=Path, default="/sata/rriochet/intphys2019/traintest"
    )
    parser.add_argument("--hw", type=int, default=288)
    parser.add_argument("--N_o", type=int, default=6, help="number of balls")
    parser.add_argument("--video_slice", type=parse_slice, default=slice(None))
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    return args


def main():
    args = parse_args()
    print(args)
    loader = make_loader(args, "train")
    model = make_model(MaskRCNN, args, args.gpu, args.data_parallel, args.load)
    model.nn_module.eval()
    for i, batch in enumerate(loader):
        res_list = []
        preds = model(batch[0])
        for (image_tensor, pred_dict) in zip(batch[0], preds):
            image_array = image_tensor.numpy().transpose((1, 2, 0))
            im = ti(image_array)
            draw = ImageDraw.Draw(im)
            boxes = pred_dict["boxes"].detach().cpu().numpy()
            labels = pred_dict["labels"].detach().cpu().numpy()
            scores = pred_dict["scores"].detach().cpu().numpy()
            for box, label, score in zip(boxes, labels, scores):
                if score > args.cutoff:
                    draw.rectangle(box.tolist(), outline=tuple(COLOR_LIST[label]))
            res_list.append(np.array(im))
        im = tis(np.array(res_list).transpose((0, 3, 1, 2)))
        im.save(f"inference/{i:04d}.png")


if __name__ == "__main__":
    main()
