import hashlib
import math
import os
from io import BytesIO
from itertools import cycle

import cv2
import numpy as np
import torch
from colour import Color
from PIL import Image

COLOR_LIST = [
    [int(c * 255) for c in Color(name).rgb][:3]
    for name in [
        "red",
        "green",
        "yellow",
        "orange",
        "turquoise",
        "blue",
        "firebrick",
        "gray",
        "magenta",
        "brown",
        "cyan",
        "purple",
        "wheat",
        "lightsalmon",
        "palevioletred",
        "darkkhaki",
        "thistle",
        "darkblue",
        "navy",
        "cornsilk",
        "sandybrown",
        "goldenrod",
        "azure",
        "beige",
        "oldlace",
        "slategray",
        "springgreen",
    ]
]


def resize(array, hw):
    return cv2.resize(array, (hw,) * 2, interpolation=cv2.INTER_NEAREST)


def image2mask(arr, color_list=COLOR_LIST):
    res = np.zeros(arr.shape[:2])
    for i, c in enumerate(color_list):
        mask = (arr == np.array(c)[:3]).all(axis=2)
        res += (i + 1) * mask
    return res


def binmask2image(arr, color_list=COLOR_LIST):
    im = np.zeros((arr.shape[1], arr.shape[2], 3))
    for m, c in zip(arr, cycle([[0, 0, 0]] + color_list)):
        im += np.repeat(np.expand_dims(m, axis=-1), 3, axis=-1) * np.array(c)
    return im.astype("uint8")


def mask2image(arr, color_list=COLOR_LIST):
    im = np.zeros((*arr.shape, 3))
    for i, c in enumerate(color_list):
        im += np.repeat(np.expand_dims(arr == (i + 1), axis=-1), 3, axis=-1) * np.array(
            c
        )
    return im.astype("uint8")


def process_multi_channel(arr):
    n_channel, i_channel = np.min(arr.shape), np.argmin(arr.shape)
    if n_channel == 1:
        return arr[i_channel]
    if n_channel == 3:
        return arr.transpose((1, 2, 0)) if i_channel == 0 else arr
    if n_channel != 3:
        if i_channel == 2:
            arr = arr.transpose((2, 0, 1))
        return binmask2image(arr)


def _normalize(array):
    array = array.copy()
    u = np.unique(array)
    q1, q3 = np.percentile(u, 25), np.percentile(u, 75)
    array = (array - (q1 + q3) / 2) / (q3 - q1) / 1.7  # 1.7 scales contrast
    array = (array * 255 + 127).clip(0, 255)
    return array


def _array2image(arr, normalize=None):
    assert len(arr.shape) in [2, 3]
    if len(arr.shape) == 3:
        arr = process_multi_channel(arr)
    if len(arr.shape) == 2:
        if arr.dtype is not np.dtype("float32"):
            arr = mask2image(arr)
    if normalize is None:
        # if arr.max() > 255 or arr.min() < 0:
        arr = _normalize(arr)
    elif normalize:
        arr = normalize(arr)
    image = Image.fromarray(arr.astype("uint8")).convert("RGB")
    return image


def array2image(arr, normalize=None):
    assert len(arr.shape) in [2, 3, 4]
    if len(arr.shape) != 4:
        return _array2image(arr, normalize)
    n = math.ceil(math.sqrt(len(arr)))
    arr = np.pad(
        arr,
        ((0, 0), (0, 0), (5, 5), (5, 5)),
        "constant",
        constant_values=len(COLOR_LIST) - 1,
    )
    array_list = [arr[i : i + n] for i in range(0, len(arr), n)]
    array_list = [np.concatenate(a, axis=2) for a in array_list]
    p = array_list[0].shape[-1] - array_list[-1].shape[-1]
    array_list[-1] = np.pad(array_list[-1], ((0, 0), (0, 0), (0, p)), "minimum")
    return _array2image(np.concatenate(array_list, axis=1), normalize)


def tensor2image(tensor, normalize=None):
    array = tensor.clone().detach().cpu().numpy()
    image = array2image(array, normalize)
    return image


def to_image_server(im):
    byteimage = BytesIO()
    try:
        im.save(byteimage, format="PNG", compress=1)
    except AttributeError:
        im.savefig(byteimage)
    im_hash = hashlib.sha256(im.tobytes()).hexdigest()
    name = os.environ["HTML_FOLDER"] + f"/{im_hash}.png"
    im.save(name)
    print(os.environ["HTML_ADRESS"] + f"/{im_hash}.png")


def ti(x, normalize=None):
    if type(x) is torch.Tensor:
        im = tensor2image(x, normalize)
    elif type(x) is np.ndarray:
        im = array2image(x, normalize)
    elif type(x) is Image.Image:
        im = x
    else:
        raise ValueError
    return im


def tis(x, normalize=None):
    im = ti(x, normalize)
    to_image_server(im)
    return im
