import numpy as np

from pyronan.model import Model


class Base(Model):
    def __init__(self, *_):
        self.min_size = self.max_size = 100
        super().__init__()

    def step(self, batch, _):
        batch = [
            {k: v.to(self.device) for k, v in x.items() if k not in ["height", "width"]}
            for x in batch
        ]
        return {"loss": 0}

    def get_image(self, _):
        return np.zeros((1, 1, self.min_size, self.max_size)).astype("uint8")
