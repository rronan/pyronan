import colorsys
import random
from itertools import cycle, zip_longest

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pyronan.utils.image import COLOR_LIST


def make_random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    colors = (colors * 255).astype(np.uint8).tolist()
    random.shuffle(colors)
    return colors


def draw_box(draw, xy, color, width=5, alpha=128):
    for z in range(width):
        x = [b + a for b, a in zip(xy, [-z, -z, z, z])]
        draw.rectangle(x, fill=None, outline=tuple(color + [alpha]))


def draw_label(draw, label, position, color, score=None, label_map={}, fontsize=15):
    text = label_map.get(label, f"{label}")
    if score is not None:
        text += f"{score:.0%}"
    font = ImageFont.truetype("arial.ttf", fontsize)
    ts = draw.textsize(text, font)
    xy = [
        (position[0], max(0, position[1] - ts[1])),
        (position[0] + ts[0], max(0, position[1] - ts[1]) + ts[1]),
    ]
    draw.rectangle(xy, fill=tuple(color))
    draw.text(xy[0], text, fill=((np.mean(color) < 128) * 255,) * 4, font=font)


def draw_keypoint(draw, xy, color, size):
    box = [xy[0] - size, xy[1] - size, xy[0] + size, xy[1] + size]
    outline = ((np.mean(color) < 128) * 255,) * 4
    draw.pieslice(box, 0, 360, fill=tuple(color), outline=outline)


def draw_keypoints(draw, keypoints, color, adjacency_matrix=None, size=3):
    for xy in keypoints:
        draw_keypoint(draw, xy, color, size)


def draw_mask_polygon(draw, mask, color):
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1)
    )
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    # either this:
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        draw.polygon([tuple(x[0]) for x in polygon], outline=(255, 255, 255))
    # or that:
    # hull = cv2.convexHull(np.concatenate(polygons)[:, 0]).reshape(-1, 2)
    # if len(hull) > 1:
    #     draw.polygon([tuple(x) for x in hull], outline=(255, 255, 255))


def draw_binmask(image, binmask, color):
    rgbmask = np.tile(np.array(color)[np.newaxis, np.newaxis], image.size[::-1] + (1,))
    rgbmask = Image.fromarray(rgbmask.astype(np.uint8))
    image.paste(
        Image.blend(image, rgbmask, 0.5), (0, 0), Image.fromarray(binmask > 0.5)
    )


def draw_mask(image, mask, color_list=COLOR_LIST):

    for c, v in zip(color_list, np.unique(mask)):
        binmask = mask == v
        draw_binmask(image, binmask, c)


def draw_detection(image, detection, colors="random", label_map={}):
    draw = ImageDraw.Draw(image, "RGBA")
    n = len(detection["boxes"])
    if colors == "random":
        colors = make_random_colors(n)
    elif colors == "COLOR_LIST":
        colors = cycle(COLOR_LIST)
    keys = list(detection.keys())
    for *value, color in zip_longest(*(detection[k] for k in keys), colors):
        instance = dict(zip(keys, value))
        draw_box(draw, instance["boxes"], color)
        if "labels" in instance:
            draw_label(
                draw,
                instance["labels"],
                instance["boxes"][:2],
                color,
                score=instance.get("scores"),
                label_map=label_map,
            )
        if "masks" in instance:
            draw_binmask(image, instance["masks"], color)
            draw_mask_polygon(draw, instance["masks"], color)
        if "keypoints" in instance:
            draw_keypoints(draw, instance["keypoints"], color)
    return np.array(image)


def gather_batch(image_list):
    mh = max([image.shape[0] for image in image_list])
    res = [np.pad(x, ((0, mh - x.shape[0]), (0, 0), (0, 0))) for x in image_list]
    res = np.concatenate(res, axis=1)
    return res


def draw_detection_batched(
    image_batch, detection_batch, colors_batch="random", label_map={}
):
    image_list = []
    if type(colors_batch) is str:
        colors_batch = [colors_batch] * len(image_batch)
    for image, detection, colors in zip(image_batch, detection_batch, colors_batch):
        image_list.append(draw_detection(image, detection, colors, label_map))
    res = gather_batch(image_list)
    return res


########################################################################################
def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tobytes()).convert("RGB")


def fig2arr(fig):
    img = fig2img(fig)
    plt.close(fig)
    return np.array(img)
