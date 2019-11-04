# pyronan
Framework and utilities for training models with Pytorch


## Install 

```
git clone git@github.com:rronan/pyronan.git
pip install -e pyronan
```

## Example

Define ```MyModel``` as subclass of ```pyronan.model.Model```

Define ```MyDataset``` as subclasses of ```pyronan.dataset.Dataset```

Minimal script:

```
from argparse import ArgumentParser
from pathlib import Path

from pyronan.utils.train import checkpoint, make_loader, trainer

parser = ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default="path to checkpoints")
parser.add_argument("--bsz", type=int, default=16, help="batch size")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--save_best", action='store_true')
parser.add_argument("--save_all", action="store_true")
parser.add_argument("--name", default=None)
args = parser.parse_args(argv)
args.name = append_timestamp(args.name)
args.checkpoint /= args.name


dataset_path = "pyronan.examples.mnist.dataset"
loader_dict = {
    set_: make_loader(MyDataset, args, set_, args.bsz, args.num_workers, args.pin_memory)
    for set_ in ["train", "val"]
}
model = MyModel()
checkpoint_func = partial(checkpoint, model=model, args=args)
trainer(model, loader_dict, args.n_epochs, checkpoint_func)
```

