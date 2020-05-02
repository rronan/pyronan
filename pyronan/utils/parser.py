from argparse import ArgumentParser

from path import Path

from pyronan.utils.misc import append_timestamp

parser_base = ArgumentParser(add_help=False)
parser_base.add_argument("--seed", type=int, default=0)
parser_base.add_argument("--name", type=append_timestamp, default="")
parser_base.add_argument("--checkpoint", type=Path)
parser_base.add_argument("--loglevel", type=int, default=20)
