import argparse
import itertools
import time
from copy import copy
from pathlib import Path
from pydoc import locate
import os

import yaml
from dask.distributed import Client
from dask_jobqueue import SGECluster
from pyronan.utils.html_results import make_html
from pyronan.utils.misc import append_timestamp

# dask-submit <remote-client-address>:<port> distribute.py


def init_cluster(args):
    cluster = SGECluster(
        queue=args.queue,
        cores=1,
        # processes=1,
        local_directory=args.logs_dir,
        memory=f"{args.mem_req}GB",
        resource_spec=f"h_vmem={args.h_vmem},mem_req={args.mem_req}",
        interface="ib0",
    )
    cluster.scale(jobs=args.jobs)
    client = Client(cluster)
    return client


def make_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f)
    if "name" not in config:
        config["name"] = config_path.stem
    return config


def update_opt(opt, dict_):
    opt_copy = copy(opt)
    for k, v in dict_.items():
        vars(opt_copy)[k] = v
    return opt_copy


def make_args_list(config):
    res = []
    config["name"] = append_timestamp(config["name"])
    baseopt = update_opt(locate(config["parser"])([]), config["args"])
    for sweep in config["grids"]:
        for values in itertools.product(*sweep.values()):
            opt = update_opt(baseopt, sweep)
            opt.name = "_".join([config["name"], str(len(res))])
            res.append(opt)
    return res


def submit(client, config):
    func = locate(config["function"])
    args_list = make_args_list(config)
    res = []
    for opt in args_list:
        res.append({"future": client.submit(func, opt), "opt": opt})
    return res


def is_running(job_list):
    for job in job_list:
        if job["future"].status != "finished":
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, default="sweep.yaml")
    parser.add_argument("--exclude_nodes", nargs="+", default=[])
    parser.add_argument(
        "--logs_dir", type=Path, default=os.environ.get("PYRONAN_LOGS_DIR")
    )
    parser.add_argument(
        "--html_dir", type=Path, default=os.environ.get("PYRONAN_HTML_DIR")
    )
    parser.add_argument("--queue", default="gaia.q,zeus.q,titan.q,chronos.q")
    parser.add_argument("--mem_req", type=int, default=100)
    parser.add_argument("--h_vmem", type=int, default=200000)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--wait", type=int, default=60)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    client = init_cluster(args)
    config = make_config(args.config_path)
    job_list = submit(client, config)
    while is_running(job_list):
        time.sleep(args.wait)
        res = make_html(
            config,
            sorted([jb["opt"] for jb in job_list], key=lambda x: x.name),
            args.html_dir,
        )
        print(res, time.time())


if __name__ == "__main__":
    main()
