import argparse
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

    <b>Min loss on val: {global_min} ({global_min_name})</b>

    {body}

</body>
</html>
"""
block_template = """

    <hr>
    <p>{name}: {info}</p>
    <p>Min loss on train: {min_train} (epoch: {argmin_train})</p>
    <p>Min loss on val: {min_val} (epoch: {argmin_val})</p>
    <img src="figures/{name}_loss.png" style="margin:1px; padding:1px;" width="512"> <img src="figures/{name}_lr.png" style="margin:1px; padding:1px;" width="512">

"""


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


def make_body_chunk(checkpoint, info, global_min, global_min_name, key, outdir):
    name = Path(checkpoint)
    success, min_train, argmin_train, min_val, argmin_val = draw_curves(
        checkpoint / "log.json", outdir / "figures" / name, key
    )
    if success:
        if min_val < global_min:
            global_min = min_val
            global_min_name = checkpoint.name
        block = block_template.format(
            name=checkpoint.name,
            info=info,
            min_train=min_train,
            argmin_train=argmin_train,
            min_val=min_val,
            argmin_val=argmin_val,
        )
        return (block, global_min, global_min_name)


def make_html(config, checkpoint_list, params_list, key="loss", outdir=None):
    tempdir = None
    body = ""
    global_min = math.inf
    global_min_name = "Null"
    for checkpoint, params in zip(checkpoint_list, params_list):
        name = checkpoint.name
        body_chunk, global_min, global_min_name = make_body_chunk(
            name, params, global_min, global_min_name, key, tempdir
        )
        image_line = ""
        for image in checkpoint.files("*.png"):
            out_path = f"figures/{name}_{image}"
            image.copyfile(tempdir / out_path)
            image_line += (
                f'<img src="{out_path}" style="margin:1px; padding:1px;" width="512">'
            )
        body += body_chunk + "\n<p>" + image_line + "\n</p>"
    html = html_template.format(
        config=config, global_min=global_min, global_min_name=global_min_name, body=body
    )
    with open(tempdir / "index.html", "w") as f:
        f.write(html)
    if outdir is None:
        outdir = os.environ["PYRONAN_HTML_DIR"]
    # TODO mv temp to outdir


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
    checkpoint_list = None
    params_list = None
    make_html(config, checkpoint_list, params_list, args.key, args.outdir)


if __name__ == "__main__":
    main()
