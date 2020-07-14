import argparse
import imp
import logging
from pathlib import Path
from pydoc import locate

import submitit

from pyronan.distribute.utils import make_opt_list, parser_distribute

logging.basicConfig(level=logging.INFO)


def init_executor(executor, args):
    log_folder = f"{args.log_dir}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=args.timeout_hour * 60,
        slurm_partition="dev",
        gpus_per_node=args.ngpus,
    )
    return executor


def submit(executor, config, merge_names):
    opt_list = make_opt_list(config, merge_names)
    print(*opt_list, sep="\n************\n\n")
    jobs = executor.map_array(config.FUNCTION, opt_list)
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(parents=[parser_distribute])
    parser.add_argument("--exclude_nodes", nargs="+", default=[])
    parser.add_argument("--queue", default="gaia.q,zeus.q,titan.q,chronos.q")
    parser.add_argument("--mem_req", type=int, default=32)
    parser.add_argument("--h_vmem", type=int, default=200000)
    parser.add_argument("--timeout_hour", type=int, default=48)
    parser.add_argument("--to_source", default=None)
    parser.add_argument(
        "--export_var", nargs="*", default=["PYTHONPATH", "TORCH_MODEL_ZOO"]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.info(args)
    config = imp.load_source("config", args.config_file)
    if not hasattr(config, "NAME"):
        config.NAME = args.config_file.stem
    print(config.NAME)
    executor = init_executor(config.NAME, args)
    job_list = submit(executor, config, args.merge_names)
    logging.info([job.job_id for job in job_list])
    print(*[job.result() for job in job_list], sep="\n************\n\n")


if __name__ == "__main__":
    main()
