import json
from itertools import product

from pyronan.utils import train
from pyronan.utils.misc import to_namespace


def test_make_model():
    for p, model in []:
        for gpu, data_parallel in product([True, False], [True, False]):
            with open(p, "r") as f:
                args = to_namespace(json.load(f))
            train.make_model(model, args, args.load, gpu, data_parallel)


def test_make_loader():
    for p, dataset in []:
        for set_, bsz, num_workers, pin_memory, shuffle in product(
            ["train", "val"], [1, 10], [0, 4], [True, False], [True, False]
        ):
            with open(p, "r") as f:
                args = to_namespace(json.load(f))
            train.make_loader(
                dataset, args, set_, bsz, num_workers, pin_memory, shuffle
            )


def test_process_batch():
    for args_path, model_path, dataset_path in zip([], [], []):
        with open(args_path, "r") as f:
            args = to_namespace(json.load(f))
        model = train.make_model(
            model_path, args, args.load, args.gpu, args.data_parallel
        )
        loader = train.make_loader(dataset_path, args, "train", 10, 0, False)
        loss = {}
        for j, batch in enumerate(loader):
            train.process_batch(model, batch, loss, "train", j)
            if j > 10:
                break


def test_process_epoch():
    pass


def test_trainer():
    pass


def checkpoint():
    pass
