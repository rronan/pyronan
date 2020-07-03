import argparse
import logging
from pathlib import Path
from pydoc import locate

import submitit
from dask.distributed import Client, as_completed
from dask_jobqueue import SGECluster
from pyronan.distribute.utils import make_config, make_opt_list, parser_distribute

logging.basicConfig(level=logging.INFO)


def init_executor(executor, args):
    log_folder = f"{args.log_dir}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=4, slurm_partition="dev", gpus_per_node=args.ngpus
    )
    return executor


def submit(executor, config, merge_names):
    opt_list = make_opt_list(config, merge_names)
    print(opt_list)
    func = locate(config["function"])
    jobs = executor.map_array(func, opt_list)
    return jobs


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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.info(args)
    config = make_config(args.config_path)
    print(config["name"])
    executor = init_executor(config["name"], args)
    jobs = submit(executor, config, args.merge_names)
    logging.info(jobs.job_id)
    output = jobs.result()
    print(output)


if __name__ == "__main__":
    main()
