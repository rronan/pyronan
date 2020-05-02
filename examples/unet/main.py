from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pyronan.dataset import make_loader
from pyronan.examples.unet.dataset import Dataset
from pyronan.examples.unet.model import EfficientNet_unet
from pyronan.model import make_model, parser_model
from pyronan.train import Trainer, parser_train
from pyronan.utils.misc import parse_slice, parser_base, set_seed


def parse_args(argv=None):
    parser = ArgumentParser(parents=[parser_base, parser_model, parser_train])
    parser.add_argument("task", nargs="+", choices=["train", "inference"])
    parser.add_argument("--model", default="EfficientNet_unet", help="model to train")
    parser.add_argument("--b", type=int, default=3)
    parser.add_argument(
        "--input_prefix", type=Path, default="/sata/rriochet/intphys2019/test"
    )
    parser.add_argument("--select_features", type=Path, default=None)
    parser.add_argument("--clip", type=Path, default=None)
    parser.add_argument(
        "--item_list", nargs="+", default=["image_array", "semantic_mask"]
    )
    parser.add_argument("--hw", type=int, default=128)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--nc_out", type=int, default=5, help="number of channels of the target data"
    )
    parser.add_argument("--video_slice", type=parse_slice, default=slice(None))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "/sequoia/data1/rriochet/Positional_Reconstruction_Network/datasets/metadata.json"
        ),
    )
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--movement", default=None)
    parser.add_argument("--nobjects", type=int, default=None)
    parser.add_argument("--block", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args(argv)
    args.checkpoint.mkdir(exist_ok=True)
    args.checkpoint = args.checkpoint / args.name
    type_dict = {x.dest: x.type for x in parser._actions}
    return args, type_dict


def train(args):
    set_seed(args.seed, args.gpu)
    loader_dict = {
        set_: make_loader(
            Dataset, args, set_, args.bsz, args.num_workers, args.pin_memory
        )
        for set_ in ["train", "val"]
    }
    model = make_model(EfficientNet_unet, args, args.load, args.gpu, args.data_parallel)
    trainer = Trainer(model, args)
    trainer.fit(loader_dict, args.n_epochs)


def write(semantic_mask_batch, path_batch):
    bsz, nc, h, w = semantic_mask_batch.shape
    res_batch = np.zeros((bsz, h, w))
    for i in range(1, nc):
        res_batch += i * (semantic_mask_batch[:, i] > 0.5)
    res_batch = res_batch.astype("uint8")
    pool = Pool(len(path_batch))
    pool.starmap(cv2.imwrite, zip(path_batch, res_batch))
    pool.close()


def inference(args):
    set_seed(args.seed, args.gpu)
    loader = make_loader(
        Dataset, args, "test", args.bsz, args.num_workers, shuffle=False
    )
    model = make_model(EfficientNet_unet, args, args.load, args.gpu, args.data_parallel)
    for tensor, path_batch in tqdm(loader):
        if args.gpu:
            tensor = tensor.cuda()
        semantic_mask_batch = model(tensor).detach().cpu().numpy()
        write(semantic_mask_batch, path_batch)


def main():
    args, _ = parse_args()
    print(args)
    if "train" in args.task:
        train(args)
    if "inference" in args.task:
        inference(args)


if __name__ == "__main__":
    main()
