# Pyronan
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

from pyronan.utils.parser import parser_base
from pyronan.model import parser_model, make_model
from pyronan.train import parser_train, Trainer


parser = ArgumentParser(parents=[parser_model, parser_train, parser_base])
args = parser.parse_args()


loader_dict = {
    'train': DataLoader(MyDataset_train, args.batch_size, args.num_workers)
    'val': DataLoader(MyDataset_val, args.batch_size, args.num_workers)
}
model = MyModel()
trainer = Trainer(model, args)
trainer.fit(loader_dict, args.train_epochs)
```

