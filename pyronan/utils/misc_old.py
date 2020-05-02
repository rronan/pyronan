import argparse
import json
import pathlib
from copy import copy


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


def save_args(path, args):
    args_dict = args2dict(args)
    with open(path, "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)


def args2dict(args):
    args_copy = copy(args)
    for k, v in vars(args_copy).items():
        if type(v) is list:
            vars(args_copy)[k] = [tostring(w) for w in v]
        else:
            vars(args_copy)[k] = tostring(v)
    return vars(args_copy)


def tostring(v):
    if isinstance(v, pathlib.PosixPath):
        return str(v)
    if type(v) is slice:
        return write_slice(v)
    elif type(v) is argparse.Namespace:
        return write_namespace(v)
    return v


def write_slice(s):
    return f"{s.start if s.start is not None else ''}:{s.stop if s.stop is not None else ''}"


class to_namespace:
    def __init__(self, d):
        vars(self).update(dict([(key, value) for key, value in d.items()]))

    def __str__(self):
        return str(vars(self))


def load_args(path, type_dict):
    with open(str(path), "r") as f:
        dict_ = json.load(f)
    for k, v in dict_.items():
        if k != "lr" and type_dict[k] is not None:
            if type(v) is list:
                v = [type_dict[k](e) for e in v]
            elif v is not None:
                v = type_dict[k](v)
        dict_[k] = v
    args = to_namespace(dict_)
    return args
