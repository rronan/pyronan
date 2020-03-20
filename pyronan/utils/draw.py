import colorsys
import random
from itertools import cycle, zip_longest

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pyronan.utils.image import COLOR_LIST, ti


def make_random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    colors = (colors * 255).astype(np.uint8).tolist()
    random.shuffle(colors)
    return colors


def draw_box(draw, xy, color, width=3, alpha=200):
    for z in range(width):
        x = [b + a for b, a in zip(xy, [-z, -z, z, z])]
        draw.rectangle(x, fill=None, outline=tuple(color + [alpha]))


def draw_label(draw, label, position, color, score=None, fontsize=15, label_map={}):
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


def draw_mask(image, mask, color):
    rgbmask = np.tile(np.array(color)[np.newaxis, np.newaxis], image.size[::-1] + (1,))
    rgbmask = Image.fromarray(rgbmask.astype(np.uint8))
    image.paste(Image.blend(image, rgbmask, 0.5), (0, 0), Image.fromarray(mask > 0.5))


def draw_detection(image, target, random_colors=True, label_map={}):
    image = ti(image, False)
    draw = ImageDraw.Draw(image, "RGBA")
    n = len(target["boxes"])
    colors = make_random_colors(n) if random_colors else cycle(COLOR_LIST)
    keys = list(target.keys())
    for *value, color in zip_longest(*(target[k] for k in keys), colors):
        instance = dict(zip(keys, value))
        labels = instance["labels"]
        if labels == 0:
            continue
        boxes = instance["boxes"]
        scores = instance.get("scores")
        draw_box(draw, boxes, color)
        draw_label(draw, labels, boxes[:][:2], color, scores, label_map=label_map)
        if "masks" in instance:
            draw_mask(image, instance["masks"], color)
            draw_mask_polygon(draw, instance["masks"], color)
        if "keypoints" in instance:
            draw_keypoints(draw, instance["keypoints"], color)
    return image


def draw_detection_batched(images, targets, random_colors=True, label_map={}):
    image_list = [
        draw_detection(x, y, random_colors, label_map) for x, y in zip(images, targets)
    ]
    res = ti([np.array(image) for image in image_list])
    return res


########################################################################################
def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf


def _draw_detection(args):
    image_array, boxes, labels = args
    im = ti(image_array)
    draw = ImageDraw.Draw(im, "RGBA")
    width = 5
    for box, label in zip(boxes, labels):
        if label != 0:
            for z in range(width):
                x = [b + a for b, a in zip(box, [z, z, -z, -z])]
                draw.rectangle(x, fill=None, outline=tuple(COLOR_LIST[label] + [128]))
    # im = im.convert("RGB")
    return np.array(im)


def draw_detection_batch(images, targets, predictions, cutoff=0):
    image_list, true_boxes, true_labels, pred_boxes, pred_labels = [], [], [], [], []
    for image, target, prediction in zip(images, targets, predictions):
        image_list.append(image.numpy().transpose((1, 2, 0)))
        box, label = [], []
        if len(target) > 0:
            c = target["labels"].detach().cpu().numpy() != 0
            box = np.array([b.numpy() for b in target["boxes"]])[c].tolist()
            label = target["labels"].numpy()[c].tolist()
        true_boxes.append(box)
        true_labels.append(label)
        c = prediction["scores"].detach().cpu().numpy() > cutoff
        c *= prediction["labels"].detach().cpu().numpy() != 0
        pred_boxes_instance = [b.detach().cpu().numpy() for b in prediction["boxes"]]
        pred_boxes.append(np.array(pred_boxes_instance)[c].tolist())
        pred_labels.append(prediction["labels"].detach().cpu().numpy()[c].tolist())
    true = list(map(_draw_detection, zip(image_list, true_boxes, true_labels)))
    pred = list(map(_draw_detection, zip(image_list, pred_boxes, pred_labels)))
    max_size = [max([true[0].shape[i], true[1].shape[i]]) for i in [0, 1]]
    p = [[max_size[i] - x.shape[i] for i in [0, 1]] for x in true]
    true = [
        np.pad(true[i], pad_width=((0, p[i][0]), (0, p[i][1]), (0, 0))) for i in [0, 1]
    ]
    true = np.array(true)
    pred = [
        np.pad(pred[i], pad_width=((0, p[i][0]), (0, p[i][1]), (0, 0))) for i in [0, 1]
    ]
    pred = np.array(pred)
    res = np.concatenate([true, pred], axis=1).transpose((0, 3, 1, 2))
    return (res * 255).astype("uint8")


########################################################################################
