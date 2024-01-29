"""Common Augmentations"""

import random

from PIL import ImageFilter, ImageOps


class GaussianBlur:
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, prob: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ) -> None:
        self.prob = prob
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization:
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            return ImageOps.solarize(img)
        else:
            return img
