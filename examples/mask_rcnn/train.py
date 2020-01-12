from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.utils.data

from dataset import Dataset_custom
from examples.mask_rcnn.model import MaskRCNN
from pyronan.model import parser_optim
from pyronan.train import make_model, parser_train, trainer
from pyronan.utils.misc import Callback, parse_slice
from vision.references.detection import group_by_aspect_ratio


def parse_args(argv=None):
    parser = ArgumentParser(parents=[parser_optim, parser_train])
    parser.set_defaults(
        checkpoint="/sequoia/data1/rriochet/pyronan/examples/mask_rcnn/checkpoints",
        name="",
        bsz=8,
        num_workers=4,
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
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint /= args.name
    return args


def collate_fn(batch):
    return tuple(zip(*batch))


def make_loader(args, set_):
    dataset = Dataset_custom(args, set_)
    sampler = torch.utils.data.RandomSampler(dataset)
    if args.aspect_ratio_group_factor >= 0:
        group_ids = group_by_aspect_ratio.create_aspect_ratio_groups(
            dataset, k=args.aspect_ratio_group_factor
        )
        batch_sampler = group_by_aspect_ratio.GroupedBatchSampler(
            sampler, group_ids, args.bsz
        )
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.bsz, drop_last=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return loader


def main():
    args = parse_args()
    print(args)
    loader_dict = {set_: make_loader(args, set_) for set_ in ["train", "val"]}
    model = make_model(MaskRCNN, args, None, args.gpu, args.data_parallel)
    trainer(model, loader_dict, args.n_epochs, Callback(model, args), args.verbose)


if __name__ == "__main__":
    main()