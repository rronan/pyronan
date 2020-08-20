import csv
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("dirs")
    parser.add_argument("--logs_name", default="loss_min.json")
    parser.add_argument("--args_name", default="args.json")
    parser.add_argument("--output", "-o", default="hiplot.csv")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    return args


def make_dict_list(dirs, args_name, logs_name, verbose):
    dict_list = []
    for dir in sorted(Path(".").glob(dirs)):
        try:
            with open(dir / args_name, "r") as f:
                chkpt_args = json.load(f)
            with open(dir / logs_name, "r") as f:
                chkpt_logs = json.load(f)
            chkpt_logs_flat = {}
            for key, value in chkpt_logs.items():
                if type(value) is dict:
                    for k, v in value.items():
                        chkpt_logs_flat[f"{key}_{k}"] = v
                else:
                    chkpt_logs_flat[key] = value
            dict_list.append({**chkpt_args, **chkpt_logs_flat})
            if verbose:
                print(dir)
        except Exception as e:
            pass
    return dict_list


def remove_constant(dict_list):
    key_set = set().union(*(d.keys() for d in dict_list))
    k_vset = {key: set() for key in key_set}
    for dict_ in dict_list:
        for k in k_vset.keys():
            k_vset[k].add(dict_.get(k))
    key_list = sorted([k for k, vset in k_vset.items() if len(vset) > 1])
    res = [{k: v for k, v in dict_.items() if k in key_list} for dict_ in dict_list]
    return key_list, res


def main(argv=None):
    args = parse_args(argv)
    dict_list = make_dict_list(args.dirs, args.args_name, args.logs_name, args.verbose)
    key_list, filtered_dict_list = remove_constant(dict_list)
    with open(args.output, "w") as f:
        dict_writer = csv.DictWriter(f, key_list)
        dict_writer.writeheader()
        dict_writer.writerows(filtered_dict_list)
    return len(dict_list)


def __main__(argv=None):
    return main(argv)


if __name__ == "__main__":
    __main__()
