import argparse
import copy
import itertools
import time
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


def make_args_list(params, name):
    res = []
    timestamp = time.strftime("%y%m%d_%H%M%S")
    baseopt = locate(params["parser"]).update(params["args"])
    for sweep in params["sweep"]:
        for values in itertools.product(*sweep.values()):
            opt = copy(baseopt).update(sweep)
            opt["name"] = "_".join([timestamp, name, str(len(res))])
            res.append(opt)


def submit(args):
    with open(args.sweep_path, "r") as f:
        params = yaml.load(f)
    func = locate(params["function"])
    client = init_cluster(args)
    args_list = make_args_list(params, args.name)
    future_list = [client.submit(func, opt) for opt in args_list]
    return future_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="sweep.yaml")
    parser.add_argument("--name", default="")
    parser.add_argument("--exclude_nodes", nargs="+", default=[])
    parser.add_argument("--queue", default="gaia.q,zeus.q,titan.q,chronos.q")
    parser.add_argument("--mem_req", type=int, default=20)
    parser.add_argument("--h_vmem", type=int, default=200000)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    submit(args)


if __name__ == "__main__":
    main()
