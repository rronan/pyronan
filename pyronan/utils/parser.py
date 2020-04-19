import logging
from argparse import ArgumentParser

from path import Path

from pyronan.utils.misc import append_timestamp


def convert_verbose(level):
    conversion_table = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    return conversion_table[int(level)]


parser_base = ArgumentParser(add_help=False)
parser_base.add_argument("--seed", type=int, default=0)
parser_base.add_argument("--name", type=append_timestamp, default="")
parser_base.add_argument("--checkpoint", type=Path)
parser_base.add_argument("--verbosity", type=convert_verbose, default=1)
