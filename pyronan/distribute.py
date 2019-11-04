import argparse
import itertools
import logging
import os
import time
from copy import copy
from pathlib import Path
from pydoc import locate

import yaml
from dask.distributed import Client
from dask_jobqueue import SGECluster

from pyronan.utils.html_results import make_html
from pyronan.utils.misc import append_timestamp

logging.basicConfig(level=logging.INFO)


def init_cluster(name, args):
    resource_spec = "h_vmem={}M,mem_req={}M".format(args.h_vmem, args.mem_req)
    env_extra = [
        "#$ -e {}".format(args.log_dir or "/dev/null"),
        "#$ -o {}".format(args.log_dir or "/dev/null"),
        "#$ -pe serial {}".format(args.ngpus if args.ngpus > 0 else args.ncpus),
        "source " + args.to_source,
        "export LANG=en_US.UTF-8",
        "export LC_ALL=en_US.UTF-8",
    ]
    cluster = SGECluster(
        queue=args.queue,
        resource_spec=resource_spec,
        name=name,
        cores=args.ncpus,
        memory="{}m".format(args.mem_req),
        processes=1,
        interface="ib0",
        local_directory=args.log_dir,
        env_extra=env_extra,
    )
    cluster.start_workers(args.jobs)
    return cluster


def make_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "name" not in config:
        config["name"] = config_path.stem
    return config


def update_opt(opt, dict_):
    opt_copy = copy(opt)
    for k, v in dict_.items():
        vars(opt_copy)[k] = v
    return opt_copy


def make_opt_list(config):
    res = []
    config["name"] = append_timestamp(config["name"])
    baseopt = update_opt(locate(config["parser"])([]), config["args"])
    for sweep in config["grids"]:
        for values in itertools.product(*sweep.values()):
            opt = update_opt(baseopt, sweep)
            opt.name = "_".join([config["name"], str(len(res))])
            res.append(opt)
    return res


def submit(cluster, config):
    client = Client(cluster)
    func = locate(config["function"])

    def func(opt):
        return locate(config["function"], opt)

    opt_list = make_opt_list(config)
    res = []
    for opt in opt_list:
        res.append({"future": client.submit(func, opt), "opt": opt})
    return res


def is_running(job_list):
    for job in job_list:
        if job["future"].status in ["pending", "running"]:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, default="sweep.yaml")
    parser.add_argument("--exclude_nodes", nargs="+", default=[])
    parser.add_argument(
        "--log_dir", type=Path, default=os.environ.get("PYRONAN_LOG_DIR")
    )
    parser.add_argument(
        "--html_dir", type=Path, default=os.environ.get("PYRONAN_HTML_DIR")
    )
    parser.add_argument("--queue", default="gaia.q,zeus.q,titan.q,chronos.q")
    parser.add_argument("--mem_req", type=int, default=32)
    parser.add_argument("--h_vmem", type=int, default=1e6)
    parser.add_argument("--to_source", default="~/.zshrc")
    parser.add_argument("--ncpus", type=int, default=4)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--wait", type=int, default=60)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.info(args)
    config = make_config(args.config_path)
    cluster = init_cluster(config["name"], args)
    job_list = submit(cluster, config)
    logging.info(cluster.job_script())
    while is_running(job_list):
        time.sleep(args.wait)
        make_html(
            config,
            sorted([jb["opt"] for jb in job_list], key=lambda x: x.name),
            args.html_dir / "results",
        )


if __name__ == "__main__":
    main()
