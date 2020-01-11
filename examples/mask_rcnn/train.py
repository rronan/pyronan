from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from dataset import Dataset_custom
from examples.mask_rcnn.model import MaskRCNN
from pyronan.model import parser_optim
from pyronan.train import parser_train, trainer
from pyronan.utils.misc import append_timestamp, checkpoint, parse_slice


def parse_args(argv=None):
    parser = ArgumentParser(parents=[parser_optim, parser_train])
    parser.set_defaults(
        checkpoint="/sequoia/data1/rriochet/pyronan/examples/mask_rcnn/checkpoints",
        name="",
        bsz=8,
        optimizer="SGD",
        lr=0.0025,
    )
    parser.add_argument(
        "--pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--input_prefix", type=Path, default="/sata/rriochet/intphys2019/traintest"
    )
    parser.add_argument("--hw", type=int, default=288)
    parser.add_argument("--N_o", type=int, default=6, help="number of balls")
    parser.add_argument("--video_slice", type=parse_slice, default=slice(None))
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint /= args.name
    return args


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    args = parse_args()
    print(args)
    loader_dict = {
        "train": DataLoader(
            Dataset_custom(args, "train"),
            args.bsz,
            args.num_workers,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            Dataset_custom(args, "val"),
            args.bsz,
            args.num_workers,
            collate_fn=collate_fn,
        ),
    }
    model = MaskRCNN(args)
    if args.gpu:
        model.gpu()
    if args.data_parallel:
        model.data_parallel()
    checkpoint_func = partial(checkpoint, model=model, args=args)
    trainer(model, loader_dict, args.n_epochs, checkpoint_func, verbose=args.verbose)


if __name__ == "__main__":
    main()
