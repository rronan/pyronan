import argparse
import json
import math
import os
import shutil
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from path import Path

matplotlib.use("Agg")


html_template = """
<!-- ## Code -->
<!-- <meta charset="utf-8"/> -->
<!DOCTYPE html>
<html>
<body>

    <p>
    Config:
    {config}
    </p>

    <b>Min loss on val: {min_} ({argmin})</b>

    {body}

</body>
</html>
"""
block_template = """

    <hr>
    <p>{name}: {opt}</p>
    <p>Min loss on train: {min_train} (epoch: {argmin_train})</p>
    <p>Min loss on val: {min_val} (epoch: {argmin_val})</p>
    <img src="figures/{name}_loss.png" style="margin:1px; padding:1px;" width="512"> <img src="figures/{name}_lr.png" style="margin:1px; padding:1px;" width="512">

"""

image_template = '<img src="{image_path}" style="margin:1px; padding:1px;" width="512">'


def try_allnan(func, array):
    try:
        return func(array)
    except ValueError:
        return None


def draw_curves(log_path, fig_path, key):
    if not log_path.exists():
        print("%s not found" % log_path)
        return False, None, None, None, None
    with open(log_path, "r") as f:
        log = json.load(f)
    x_axis = np.arange(len(log))
    train_loss = np.array([x.get(f"train_{key}") for x in log], dtype=np.float)
    val_loss = np.array([x.get(f"val_{key}") for x in log], dtype=np.float)
    plt.plot(x_axis, train_loss, "b", x_axis, val_loss, "r--")
    plt.savefig(fig_path + "_loss")
    plt.clf()
    plt.semilogy(x_axis, np.array([x.get("lr") for x in log]), "grey")
    plt.savefig(fig_path + "_lr")
    plt.clf()
    return (
        True,
        try_allnan(np.min, train_loss),
        try_allnan(np.argmin, train_loss),
        try_allnan(np.nanmin, val_loss),
        try_allnan(np.nanargmin, val_loss),
    )


def make_body_chunk(opt, min_, argmin, key, outdir):
    (outdir / "figures").mkdir_p()
    success, min_train, argmin_train, min_val, argmin_val = draw_curves(
        opt.checkpoint / "log.json", outdir / "figures" / opt.checkpoint.name, key
    )
    if success:
        if min_val < min_:
            min_ = min_val
            argmin = opt.checkpoint.name
        block = block_template.format(
            name=opt.checkpoint.name,
            opt=opt,
            min_train=min_train,
            argmin_train=argmin_train,
            min_val=min_val,
            argmin_val=argmin_val,
        )
        return (block, min_, argmin)
    return "", math.inf, "Null"


def make_html(config, opt_list, outdir, key="loss"):
    with tempfile.TemporaryDirectory() as tempdir:
        body = ""
        tempdir = Path(tempdir)
        min_, argmin = math.inf, "Null"
        for opt in opt_list:
            if opt.checkpoint.exists():
                body_chunk, min_, argmin = make_body_chunk(
                    opt, min_, argmin, key, tempdir
                )
                image_line = ""
                for image in opt.checkpoint.files("*.png"):
                    dest = f"figures/{opt.checkpoint.name}_{image}"
                    shutil.copyfile(image, dest)
                    image_line += image_template(image_path=dest)
                body += body_chunk + "\n<p>" + image_line + "\n</p>"
        html = html_template.format(
            config=yaml.dump(config, indent=4), min_=min_, argmin=argmin, body=body
        )
        with open(tempdir / "index.html", "w") as f:
            f.write(html)
        if (outdir / config["name"]).exists():
            shutil.rmtree(outdir / config["name"])
        res = shutil.copytree(tempdir, outdir / config["name"])
        os.chmod(res, 0o755)
        return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--key", default="loss")
    args = parser.parse_args()
    args.outdir = args.outdir / args.sweep_path.namebase
    print(args)
    return args


def main():
    args = parse_args()
    config = None
    opt_list = None
    make_html(config, opt_list, args.key, args.outdir)


if __name__ == "__main__":
    main()
