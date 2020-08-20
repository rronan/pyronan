import argparse
import imp
import logging
from pathlib import Path
from pydoc import locate
import time

import submitit
from tqdm import tqdm

from pyronan.distribute.utils import make_opt_list, parser_distribute
from pyronan.distribute import hiplot_csv

logging.basicConfig(level=logging.INFO)


def init_executor(args):
    log_folder = f"{args.log_dir}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=args.timeout_hour * 60,
        slurm_partition=args.partition,
        gpus_per_node=args.ngpus,
    )
    return executor


def submit(executor, config, prefix, merge_names):
    opt_list = make_opt_list(config, prefix, merge_names)
    jobs = executor.map_array(config.FUNCTION, opt_list)
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(parents=[parser_distribute])
    parser.add_argument("--timeout_hour", type=int, default=72)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--partition", default="learnfair")
    parser.add_argument("--run_hiplot", action="store_true")
    parser.add_argument("--chkpt_hiplot", default="checkpoints")
    args = parser.parse_args()
    if args.run_hiplot:
        assert args.chkpt_hiplot is not None
    return args


def main():
    args = parse_args()
    logging.info(args)
    config = imp.load_source("config", str(args.config_file))
    if args.prefix is None:
        args.prefix = args.config_file.stem
    executor = init_executor(args)
    running_list = submit(executor, config, args.prefix, args.merge_names)
    n_total = len(running_list)
    done_list = []
    with tqdm() as pbar:
        while len(running_list) > 0:
            done_list = [job for job in running_list if job.done()]
            if len(done_list) > 0:
                print("Done:")
                for job in done_list:
                    print(f"cat {args.log_dir}/{job.job_id}/{job.job_id}_log.")
            for job in done_list:
                running_list.remove(job)
            if args.run_hiplot:
                dirs = f"{args.chkpt_hiplot}/*{args.prefix}*"
                csv_file = str(args.config_file.parent / f"{args.prefix}.csv")
                out_hiplot = hiplot_csv.main([dirs, "-o", csv_file])
            else:
                out_hiplot = False
            desc = f"Done: {len(done_list)}/{n_total} | Hiplot: {out_hiplot}"
            time.sleep(60)
            pbar.set_description(desc)
            pbar.update(1)


def __main__():
    main()


if __name__ == "__main__":
    main()
