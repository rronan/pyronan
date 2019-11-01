import argparse
import json
import random
import re
import time
from copy import copy
from functools import wraps

import torch


def mp_cache(mp_dict):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            k = func.__name__
            k += "_".join(map(str, args))
            k += "_".join(map(lambda k, v: f"{k}_{v}", kwargs.items()))
            if k in mp_dict:
                return mp_dict[k]
            res = func(*args, **kwargs)
            mp_dict[k] = res
            return res

        return wrapper

    return decorate


def set_seed(seed, gpu):
    random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)


def parse_slice(s):
    a_list = []
    for a in s.split(":"):
        try:
            a_list.append(int(a))
        except ValueError:
            a_list.append(None)
    while len(a_list) < 3:
        a_list.append(None)
    return slice(*a_list)


def write_slice(s):
    return f"{s.start if s.start is not None else ''}:{s.stop if s.stop is not None else ''}"


def write_namespace(s):
    out = []
    for k, v in vars(s).items():
        if type(v) is not bool or v:
            out.append("--" + k)
            if type(v) is bool:
                out.append('""')
            elif type(v) is slice:
                out.append(write_slice(v))
            else:
                out.append(str(v))
    return " ".join(out)


class to_namespace:
    def __init__(self, d):
        vars(self).update(dict([(key, value) for key, value in d.items()]))

    def __str__(self):
        return str(vars(self))


def append_timestamp(name):
    if re.search("[\d]{6}_[\d]{6}", name):
        append = ""
    else:
        append = "_" + time.strftime("%y%m%d_%H%M%S")
    return name + append


def save_args(path, args, zf=None):
    args_copy = copy(args)
    for k, v in vars(args_copy).items():
        if type(v) is slice:
            vars(args_copy)[k] = write_slice(v)
        elif type(v) is argparse.Namespace:
            vars(args_copy)[k] = write_namespace(v)
    if zf is None:
        with open(path, "w") as f:
            json.dump(vars(args_copy), f, indent=4, sort_keys=True)
    else:
        zf.writestr(
            str(path).replace("/", "\\"),
            json.dumps(vars(args_copy), indent=4, sort_keys=True),
        )