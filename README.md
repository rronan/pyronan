# pyronan
Framework and utilities for training models with Pytorch


## Install 

```
git clone git@github.com:rronan/pyronan.git
pip install -e pyronan
```

## Example

Define ```MyModel``` as subclass of ```pyronan.model.Model```

Define ```MyDataset``` as subclass of ```pyronan.dataset.Dataset```

__Minimal working example__:

```
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from pyronan.model import parser_optim 
from pyronan.train import trainer
from pyronan.train import parser_train
from pyronan.utils.misc import append_timestamp, checkpoint


parser = ArgumentParser(parents=[parser_model, parser_optim])
parser.add_argument("--name", type=append_timestamp, default='')
args = parser.parse_args()
args.checkpoint /= args.name


loader_dict = {
    'train': DataLoader(MyDataset_train, args.bsz, args.num_workers)
    'val': DataLoader(MyDataset_val, args.bsz, args.num_workers)
}
model = MyModel()
checkpoint_func = partial(checkpoint, model=model, args=args)

trainer(model, loader_dict, args.n_epochs, checkpoint_func)
```

