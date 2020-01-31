import argparse
import itertools
import logging
import os
import time
from collections import OrderedDict
from copy import copy
from pathlib import Path
from pydoc import locate

import yaml
from dask.distributed import Client
from dask_jobqueue import SGECluster

from pyronan.utils.misc import append_timestamp

logging.basicConfig(level=logging.INFO)


def init_cluster(name, args):
    resource_spec = "h_vmem={}G,mem_req={}G".format(args.h_vmem, args.mem_req)
    exclude_nodes = "&".join(["!" + x for x in args.exclude_nodes])
    if len(exclude_nodes) > 0:
        exclude_nodes = "#$ -l h=" + exclude_nodes
    env_extra = [
        "#$ -e {}".format(args.log_dir or "/dev/null"),
        "#$ -o {}".format(args.log_dir or "/dev/null"),
        "#$ -pe serial {}".format(args.ngpus if args.ngpus > 0 else args.ncpus),
        exclude_nodes,
        "source " + args.to_source if args.to_source is not None else "",
        "export LANG=en_US.UTF-8",
        "export LC_ALL=en_US.UTF-8",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
        "export DISABLE_MP_CACHE=1",
    ]
    for var in args.export_var:
        env_extra.append(f'export {var}="{os.environ[var]}"')
    cluster = SGECluster(
        queue=args.queue,
        resource_spec=resource_spec,
        walltime="720:00:00",
        name=name,
        cores=args.ncpus,
        memory="{}G".format(args.mem_req),
        processes=1,
        interface="ib0",
        local_directory=args.log_dir,
        env_extra=env_extra,
        spill_dir=args.spill_dir,
        extra=["--no-nanny"],
    )
    # cluster.adapt(maximum_jobs=args.jobs)
    cluster.scale(args.jobs)
    return cluster


def make_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "name" not in config:
        config["name"] = config_path.stem
    config["name"] = append_timestamp(config["name"], end=True)
    return config


def update_opt(opt, type_dict, dict_):
    for k, v in dict_.items():
        if type_dict[k] is not None:
            if type(v) is list:
                v = [type_dict[k](e) for e in v]
            else:
                v = type_dict[k](v)
        setattr(opt, k, v)
    return opt


def make_opt_list(config, merge_names):
    res = []
    opt, type_dict = locate(config["parser"])([])
    baseopt = update_opt(opt, type_dict, config["args"])
    for grid in config["grids"]:
        grid = OrderedDict(grid)
        for values in itertools.product(*grid.values()):
            opt = update_opt(copy(baseopt), type_dict, dict(zip(grid.keys(), values)))
            if merge_names:
                opt.name = config["name"]
            else:
                opt.name = "_".join([config["name"], f"{len(res):02d}"])
            res.append(opt)
    return res


def submit(cluster, config, merge_names, preview):
    client = Client(cluster)

    def func(opt):
        return locate(config["function"])(opt)

    opt_list = make_opt_list(config, merge_names)
    res = []
    for opt in opt_list:
        print(opt)
        if not preview:
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
    parser.add_argument("--h_vmem", type=int, default=200000)
    parser.add_argument("--to_source", default=None)
    parser.add_argument(
        "--export_var", nargs="*", default=["PYTHONPATH", "TORCH_MODEL_ZOO"]
    )
    parser.add_argument("--ncpus", type=int, default=4)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--wait", type=int, default=30)
    parser.add_argument(
        "--spill_dir", type=Path, default="/sequoia/data2/rriochet/dask", help="scratch"
    )
    parser.add_argument("--make_html", action="store_true")
    parser.add_argument("--merge_names", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.info(args)
    config = make_config(args.config_path)
    print(config["name"])
    cluster = init_cluster(config["name"], args)
    job_list = submit(cluster, config, args.merge_names, args.preview)
    logging.info(cluster.job_script())
    print(f'cat {args.log_dir}/{config["name"]}.o*')
    print(f'cat {args.log_dir}/{config["name"]}.e*')
    while is_running(job_list):
        time.sleep(args.wait)
        if args.make_html:
            from pyronan.utils.html_results import make_html

            make_html(
                config,
                sorted([jb["opt"] for jb in job_list], key=lambda x: x.name),
                args.html_dir / "results",
            )


if __name__ == "__main__":
    main()
