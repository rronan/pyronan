from argparse import ArgumentParser

import albumentations
import cv2

parser_transform = ArgumentParser(add_help=False)
parser_transform.add_argument("--augmentation_list", nargs="+", default=[])
parser_transform.add_argument("--shift_limit", type=float, default=0.2)
parser_transform.add_argument("--scale_limit", type=float, default=0.2)
parser_transform.add_argument("--rotate_limit", type=float, default=30)
parser_transform.add_argument("--elastic_alpha", type=float, default=0.1)
parser_transform.add_argument("--elastic_sigma", type=int, default=50)
parser_transform.add_argument("--elastic_alpha_affine", type=int, default=50)
parser_transform.add_argument("--brightness_limit", type=float, default=0.3)
parser_transform.add_argument("--contrast_limit", type=float, default=0.3)
parser_transform.add_argument("--grid_distortion_steps", type=int, default=4)
parser_transform.add_argument("--distort_limit", type=float, default=0.1)
parser_transform.add_argument("--cutout_max_holes", type=int, default=20)
parser_transform.add_argument("--cutout_max_width", type=int, default=20)
parser_transform.add_argument("--cutout_max_height", type=int, default=20)
parser_transform.add_argument("--p_shift_scale_rotate", type=float, default=0.5)


def make_augmentation_transforms(args):
    transforms = []
    for transform_name in args.augmentation_list:
        if transform_name == "HorizontalFlip":
            transform = albumentations.HorizontalFlip(p=0.5)
        elif transform_name == "VerticalFlip":
            transform = albumentations.VerticalFlip(p=0.5)
        elif transform_name == "ElasticTransform":
            transform = albumentations.ElasticTransform(
                alpha=args.elastic_alpha,
                sigma=args.elastic_sigma,
                alpha_affine=args.elastic_alpha_affine,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
            )
        elif transform_name == "ShiftScaleRotate":
            transform = albumentations.ShiftScaleRotate(
                shift_limit=args.shift_limit,
                scale_limit=args.scale_limit,
                rotate_limit=args.rotate_limit,
                border_mode=cv2.BORDER_CONSTANT,
                p=args.p_shift_scale_rotate,
            )
        elif transform_name == "RandomBrightnessContrast":
            transform = albumentations.RandomBrightnessContrast(
                brightness_limit=args.brightness_limit,
                contrast_limit=args.contrast_limit,
            )
        elif transform_name == "GridDistortion":
            transform = albumentations.GridDistortion(
                num_steps=args.grid_distortion_steps,
                distort_limit=args.distort_limit,
                border_mode=cv2.BORDER_CONSTANT,
            )
        elif transform_name == "Cutout":
            transform = albumentations.CoarseDropout(
                max_holes=args.cutout_max_holes,
                max_width=args.cutout_max_width,
                max_height=args.cutout_max_height,
            )
        elif transform_name == "JpegCompression":
            transform = albumentations.JpegCompression()
        else:
            raise ValueError(f"Unrecognized transform type: {transform_name}")
        transforms.append(transform)
    bboxes_params = albumentations.BboxParams("pascal_voc", label_fields=["labels"])
    return albumentations.Compose(transforms, bboxes_params)
