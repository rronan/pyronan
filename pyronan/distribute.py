import argparse
import itertools
import time
from copy import copy
from pathlib import Path
from pydoc import locate

import yaml
from dask.distributed import Client
from dask_jobqueue import SGECluster

# dask-submit <remote-client-address>:<port> gridsearch.py


def init_cluster(args):
    cluster = SGECluster(
        queue=args.queue,
        cores=1,
        processes=1,
        memory="16GB",
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
    timestamp = time.strftime("%y%m%d_%H%M%S")
    baseopt = update_opt(locate(config["parser"])([]), config["args"])
    for sweep in config["grids"]:
        for values in itertools.product(*sweep.values()):
            opt = update_opt(baseopt, sweep)
            opt.name = "_".join([timestamp, config["name"], str(len(res))])
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
        if job["future"].status == "running":
            return True
    return False


def visualize(config, job_list):
    pass  # TODO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, default="sweep.yaml")
    parser.add_argument("--exclude_nodes", nargs="+", default=[])
    parser.add_argument("--queue", default="lowgpu.q,gaia.q,zeus.q,titan.q,chronos.q")
    parser.add_argument("--mem_req", type=int, default=20)
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
        visualize(config, job_list)


if __name__ == "__main__":
    main()