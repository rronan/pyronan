from argparse import ArgumentParser
from functools import partial

from path import Path

from pyronan.dataset import Dataset
from pyronan.model import Model
from pyronan.utils.misc import append_timestamp, set_seed
from pyronan.utils.train import checkpoint, make_loader, trainer


class Mnist_dataset(Dataset):
    pass


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--item_list", nargs="+", default=[])
    parser.add_argument("--checkpoint", type=Path, default="checkpoints")
    parser.add_argument("--bsz", type=int, default=16, help="batch size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hw", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--save_best", type=int, default=1)
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Use NVIDIA GPU")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument("--clip", type=float, default=3)
    args = parser.parse_args(argv)
    args.checkpoint.mkdir_p()
    args.name = append_timestamp(args.name)
    args.checkpoint /= args.name
    return args


def train(args):
    set_seed(args.seed, args.gpu)
    dataset_path = "pyronan.examples.mnist.dataset"
    loader_dict = {
        set_: make_loader(
            dataset_path, args, set_, args.bsz, args.num_workers, args.pin_memory
        )
        for set_ in ["train", "val"]
    }
    Mnist_module = None
    model = Model(Mnist_module, None)
    checkpoint_func = partial(checkpoint, model=model, args=args)
    trainer(model, loader_dict, args.n_epochs, checkpoint_func)


def main():
    args = parse_args()
    print(args)
    train(args)


if __name__ == "__main__":
    main()
