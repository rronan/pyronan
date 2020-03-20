import numpy as np
from PIL import Image, ImageDraw

from pyronan.utils.image import COLOR_LIST


def draw_point(im, mu, color):
    draw = ImageDraw.Draw(im)
    h, w = im.size
    p = (mu[:2] + 1) * np.array([h, w]) / 2
    r = 3
    box = [p[0] - r, p[1] - r, p[0] + r, p[1] + r]
    draw.ellipse(box, fill=color, outline=(255, 255, 255))


def draw_box(im, mu, color):
    draw = ImageDraw.Draw(im)
    p = (np.array([mu[0], -mu[1]]) + 1) * np.array(im.size) / 2
    r = mu[3] * np.mean(im.size) / 2
    box = [p[0] - r, p[1] - r, p[0] + r, p[1] + r]
    draw.rectangle(box, outline=color)


def make_heatmap_array(mu, sigma, hw):
    x, y = np.meshgrid(np.linspace(-1, 1, hw[0]), np.linspace(-1, 1, hw[1]))
    g = np.exp(-0.5 * (((x - mu[0]) / sigma[0]) ** 2 + ((y - mu[1]) / sigma[1]) ** 2))
    return g


def plot_position(
    im, mu_list, box=False, std_list=None, color_list=COLOR_LIST, scale=1
):
    """
               (0, 1)
                 |
                 |
                 |
                 |
(-1, 0)  ------- 0 ------- (1, 0)
                 |
                 |
                 |
                 |
               (0, -1)
    """
    if std_list is None:
        std_list = [[1 / 64.0, 1 / 64.0] for _ in mu_list]
    hw = [int(hw * scale) for hw in im.size]
    im = im.resize(hw)
    for mu, std, color in zip(mu_list, std_list, map(tuple, color_list)):
        if box:
            draw_box(im, mu, color)
        yx = np.array([mu[0], -mu[1]])
        std = np.array([std[0], std[1]])
        color_array = np.tile(np.array(color)[np.newaxis, np.newaxis], (*hw, 1))
        color_image = Image.fromarray(color_array.astype("uint8"))
        heatmap_array = make_heatmap_array(yx[:2], std[:2], hw)
        heatmap_image = Image.fromarray((heatmap_array * 255).astype("uint8"))
        im.paste(color_image, (0, 0), heatmap_image)
    return im


def draw_point(im, mu, color):
    draw = ImageDraw.Draw(im)
    h, w = im.size
    p = (mu[:2] + 1) * np.array([h, w]) / 2
    r = 3
    box = [p[0] - r, p[1] - r, p[0] + r, p[1] + r]
    draw.ellipse(box, fill=color, outline=(255, 255, 255))


def draw_box(im, mu, color):
    draw = ImageDraw.Draw(im)
    p = (np.array([mu[0], -mu[1]]) + 1) * np.array(im.size) / 2
    r = mu[3] * np.mean(im.size) / 2
    box = [p[0] - r, p[1] - r, p[0] + r, p[1] + r]
    draw.rectangle(box, outline=color)


def make_heatmap_array(mu, sigma, hw):
    x, y = np.meshgrid(np.linspace(-1, 1, hw[0]), np.linspace(-1, 1, hw[1]))
    g = np.exp(-0.5 * (((x - mu[0]) / sigma[0]) ** 2 + ((y - mu[1]) / sigma[1]) ** 2))
    return g


def plot_position(
    im, mu_list, box=False, std_list=None, color_list=COLOR_LIST, scale=1
):
    """
               (0, 1)
                 |
                 |
                 |
                 |
(-1, 0)  ------- 0 ------- (1, 0)
                 |
                 |
                 |
                 |
               (0, -1)
    """
    if std_list is None:
        std_list = [[1 / 64.0, 1 / 64.0] for _ in mu_list]
    hw = [int(hw * scale) for hw in im.size]
    im = im.resize(hw)
    for mu, std, color in zip(mu_list, std_list, map(tuple, color_list)):
        if box:
            draw_box(im, mu, color)
        yx = np.array([mu[0], -mu[1]])
        std = np.array([std[0], std[1]])
        color_array = np.tile(np.array(color)[np.newaxis, np.newaxis], (*hw, 1))
        color_image = Image.fromarray(color_array.astype("uint8"))
        heatmap_array = make_heatmap_array(yx[:2], std[:2], hw)
        heatmap_image = Image.fromarray((heatmap_array * 255).astype("uint8"))
        im.paste(color_image, (0, 0), heatmap_image)
    return im
