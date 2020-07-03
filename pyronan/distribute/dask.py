import argparse
import logging
import os
import traceback
from pathlib import Path

from dask.distributed import Client, as_completed
from dask_jobqueue import SGECluster
from pyronan.distribute.utils import make_config, make_opt_list, parser_distribute

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
        "export TORCH_HOME=/sequoia/data1/rriochet/.torch",
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


def submit(cluster, config, merge_names):
    client = Client(cluster)
    opt_list = make_opt_list(config, merge_names)
    logging.debug(opt_list)
    future_list = client.map(config.FUNCTION, opt_list)
    return opt_list, future_list


def parse_args():
    parser = argparse.ArgumentParser(parents=[parser_distribute])
    parser.add_argument("--exclude_nodes", nargs="+", default=[])
    parser.add_argument("--queue", default="gaia.q,zeus.q,titan.q,chronos.q")
    parser.add_argument("--mem_req", type=int, default=32)
    parser.add_argument("--h_vmem", type=int, default=200000)
    parser.add_argument("--to_source", default=None)
    parser.add_argument(
        "--export_var", nargs="*", default=["PYTHONPATH", "TORCH_MODEL_ZOO"]
    )
    parser.add_argument(
        "--spill_dir", type=Path, default="/sequoia/data2/rriochet/dask", help="scratch"
    )
    args = parser.parse_args()
    return args


def log(opt, future):
    print("*" * 89, "\n", opt)
    exception = future.exception()
    traceback.print_exception(type(exception), exception, future.traceback())


def main():
    args = parse_args()
    logging.info(args)
    config = make_config(args.config_path)
    print(config.NAME)
    cluster = init_cluster(config.NAME, args)
    opt_list, future_list = submit(cluster, config, args.merge_names)
    logging.info(cluster.job_script())
    print(f"cat {args.log_dir}/{config.NAME}.o*")
    print(f"cat {args.log_dir}/{config.NAME}.e*")
    try:
        c = -1  # defining variable to be use in the finally
        for c, future in enumerate(as_completed(future_list)):
            log(opt_list[c], future)
    except KeyboardInterrupt:
        print("*" * 89, "\n\nKeyboardInterrupt\n\n", "*" * 89)
        for future in future_list:
            future.cancel()
    finally:
        for opt, future in zip(opt_list[c + 1 :], future_list[c + 1 :]):
            log(opt, future)


if __name__ == "__main__":
    main()
