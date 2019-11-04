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

Minimal working example:

```
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from pyronan.utils.train import checkpoint, trainer
from pyronan.utils.misc import append_timestamp

parser = ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default="path to checkpoints")
parser.add_argument("--bsz", type=int, default=16, help="batch size")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--name", default='')
args = parser.parse_args(argv)
args.name = append_timestamp(args.name)
args.checkpoint /= args.name

loader_dict = {
    'train': DataLoader(MyDataset_train, args.bsz, args.num_workers)
    'val': DataLoader(MyDataset_val, args.bsz, args.num_workers)
}
model = MyModel()
checkpoint_func = partial(checkpoint, model=model, args=args)
trainer(model, loader_dict, args.n_epochs, checkpoint_func)
```

