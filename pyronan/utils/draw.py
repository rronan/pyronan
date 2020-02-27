import numpy as np
from PIL import Image, ImageDraw

from pyronan.utils.image import COLOR_LIST, ti


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


def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_detection(args):
    image_array, boxes, labels = args
    im = ti(image_array)
    draw = ImageDraw.Draw(im)
    for box, label in zip(boxes, labels):
        if label != 0:
            draw.rectangle(box, outline=tuple(COLOR_LIST[label]))
            box_inner = [b + a for b, a in zip(box, [1, 1, -1, -1])]
            draw.rectangle(box_inner, fill=None, outline=tuple(COLOR_LIST[label]))
    return np.array(im)


def draw_detection_batch(images, targets, predictions, cutoff=0):
    image_list, true_boxes, true_labels, pred_boxes, pred_labels = [], [], [], [], []
    for image, target, prediction in zip(images, targets, predictions):
        image_list.append(image.numpy().transpose((1, 2, 0)))
        c = target["labels"].detach().cpu().numpy() != 0
        true_boxes.append(np.array([b.numpy() for b in target["boxes"]])[c].tolist())
        true_labels.append(target["labels"].numpy()[c].tolist())
        c = prediction["scores"].detach().cpu().numpy() > cutoff
        c *= prediction["labels"].detach().cpu().numpy() != 0
        pred_boxes_instance = [b.detach().cpu().numpy() for b in prediction["boxes"]]
        pred_boxes.append(np.array(pred_boxes_instance)[c].tolist())
        pred_labels.append(prediction["labels"].detach().cpu().numpy()[c].tolist())
    true = list(map(draw_detection, zip(image_list, true_boxes, true_labels)))
    pred = list(map(draw_detection, zip(image_list, pred_boxes, pred_labels)))
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
