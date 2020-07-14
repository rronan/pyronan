import argparse
import itertools
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from pydoc import locate

import yaml

from pyronan.utils.misc import append_timestamp

parser_distribute = argparse.ArgumentParser(add_help=False)
parser_distribute.add_argument("config_file", default="sweep.yaml")
parser_distribute.add_argument(
    "--log_dir", type=Path, default=os.environ.get("PYRONAN_LOG_DIR")
)
parser_distribute.add_argument("--ncpus", type=int, default=4)
parser_distribute.add_argument("--ngpus", type=int, default=1)
parser_distribute.add_argument("--jobs", type=int, default=1)
parser_distribute.add_argument("--merge_names", action="store_true")


def make_opt_list(config, merge_names):
    res = []
    name = append_timestamp(config.NAME)
    for grid in config.GRID_LIST:
        grid = OrderedDict(grid)
        for values in itertools.product(*grid.values()):
            opt = deepcopy(config.BASE_ARGS)
            for k, v in zip(grid.keys(), values):
                setattr(opt, k, v)
            if merge_names:
                opt.name = name
            else:
                opt.name = "_".join([name, f"{len(res):02d}"])
            res.append(opt)
    return res
