import numpy as np
from PIL import ImageDraw

from pyronan.utils.image import COLOR_LIST, ti

# torch.multiprocessing.set_sharing_strategy("file_system")


def is_backbone_grad(lr):
    for kv in lr:
        if type(kv) is float:
            return True
        else:
            if kv[0] == "backbone":
                return True
    return False


def draw(args):
    image_array, boxes, labels = args
    im = ti(image_array)
    draw = ImageDraw.Draw(im)
    for box, label in zip(boxes, labels):
        if label != 0:
            draw.rectangle(box, outline=tuple(COLOR_LIST[label]))
            box_inner = [b + a for b, a in zip(box, [1, 1, -1, -1])]
            draw.rectangle(box_inner, fill=None, outline=tuple(COLOR_LIST[label]))
    return np.array(im)


def draw_batch(images, targets, predictions, cutoff=0):
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
    true = list(map(draw, zip(image_list, true_boxes, true_labels)))
    pred = list(map(draw, zip(image_list, pred_boxes, pred_labels)))
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
