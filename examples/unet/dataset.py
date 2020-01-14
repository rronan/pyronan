from functools import wraps

import numpy as np
import torch.utils.data
import pydicom
from path import Path
import nibabel as nib
import os
from utils.data import read_dcm, read_mask, get_dcm_array, get_mask_array
from utils.level import louis_window
from utils.crop import crop_and_resize
from multiprocessing import Manager

sarcophor_data_path = Path(os.environ["SARCOPHOR_DATA"])


def mp_cache(mp_dict):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            k = func.__name__
            k += "_".join(map(str, args))
            k += "_".join(map(lambda k, v: f"{k}_{v}", kwargs.items()))
            if k in mp_dict:
                return mp_dict[k]
            res = func(*args, **kwargs)
            mp_dict[k] = res
            return res

        return wrapper

    return decorate


class Dataset(torch.utils.data.Dataset):

    shared_dict = Manager().dict()

    def __init__(
        self,
        df,
        hw=512,
        preprocessing_transforms=None,
        augmentation_transforms=None,
        net_specific_transforms=None,
        item_list=["pixel_array", "mask_array"],
    ):
        self.item_list = (
            item_list
        )  # custom list of items to get (mask, image, age ... whatever)
        self.df = df
        self.hw = hw

        # Parsing args to get preprocessing transforms
        self.preprocessing = preprocessing_transforms

        self.augmentation = False
        # Parsing args to get augmentation transforms
        if augmentation_transforms is not None:
            self.augmentation = True
            self.aug_transforms = augmentation_transforms

        self.net_specific_transforms = (
            net_specific_transforms
        )  # only range, mean and std modified by this on image,
        # to adjust to net training.

    def get_pixel_array(self, i):
        # new version, does + intercept and / 255. already
        dcm_array = get_dcm_array(read_dcm(self.df.iloc[i]))
        # + 70/255. and crop from 0 to 1 to keep relevant range
        return dcm_array
        # return np.expand_dims(dcm_array, axis=2)  # single channel image H, W, 1

    def get_mask_array(self, i):
        mask_array = get_mask_array(read_mask(self.df.iloc[i]))
        return mask_array
        # return np.expand_dims(mask_array, axis=2)  # single channel image H, W, 1

    def get_quality_weight(self, i):
        quality = self.df.iloc[i]["quality"]
        d = {"P": 3.0, "A": 2.0, "B": 1.5, "C": 0.75, "D": 0.5}
        return np.array([d[quality]])

    def get_foo(self, i):
        pass

    def __getitem__(self, i):
        # ACHTUNG: code changed to look like Malo loading for contour detecting
        data = {item: getattr(self, "get_" + item)(i) for item in self.item_list}
        row = self.df.iloc[i]
        if i not in Dataset.shared_dict:
            Dataset.shared_dict[i] = crop_and_resize(
                data["pixel_array"],
                data["mask_array"],
                row["pixel_width"],
                row["pixel_height"],
                target_size=self.hw,
                crop_method="malo",
                square_pixel=True,
            )

        pixel_array, mask_array = Dataset.shared_dict[i]

        pixel_array = louis_window(pixel_array)

        # Now reshaping before the rest of the augmentations:
        data["pixel_array"] = np.expand_dims(pixel_array, axis=2)  # H W 1
        data["mask_array"] = np.expand_dims(mask_array, axis=2)

        # Extra preprocessing
        data_transformed = self.preprocessing(
            image=data["pixel_array"], mask=data["mask_array"], **self.df.iloc[i]
        )

        if self.augmentation:
            data_transformed = self.aug_transforms(
                image=data_transformed["image"],
                mask=data_transformed["mask"],
                **self.df.iloc[i],
            )

        # for mean, range, std
        if self.net_specific_transforms is not None:
            data_transformed = self.net_specific_transforms(
                image=data_transformed["image"],
                mask=data_transformed["mask"],
                **self.df.iloc[i],
            )

        data["pixel_array"] = data_transformed["image"]
        data["mask_array"] = data_transformed["mask"]
        if "quality_weight" in self.item_list:
            return (
                data["pixel_array"],
                data["mask_array"],
                torch.from_numpy(data["quality_weight"]).type(torch.FloatTensor),
            )
        else:
            return data["pixel_array"], data["mask_array"]

    def __len__(self):
        return len(self.df)
