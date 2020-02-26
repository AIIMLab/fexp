"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from fexp.image import clip_and_scale


class Compose(object):
    """Compose several transformations together, for instance ClipAndScale and a flip.

    Code based on torchvision: https://github.com/pytorch/vision, but got forked from these as torchvision has some
    additional dependencies.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self):
        repr_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            repr_string += '\n'
            repr_string += f'    {transform}'
        repr_string += '\n)'
        return repr_string


class ClipAndScale(object):
    """Clip input array and rescale image data.
    """
    def __init__(self, clip_range, source_interval, target_interval):
        """
        Clips image to specified range, and then linearly scales to the specified range (if given).

        In particular, the range in source interval is mapped to the target interval linearly,
        after clipping has been applied.

        - If clip_range is not set, the image is not clipped.
        - If target_interval is not set, only clipping is applied.
        - If source_interval is not set, the minimum and maximum values will be picked.

        Parameters
        ----------
        clip_range : tuple
            Range to clip input array to.
        source_interval : tuple
           If given, this denote the original minimal and maximal values.
        target_interval : tuple
            Interval to map input values to.
        """
        self.clip_range = clip_range
        self.source_interval = source_interval
        self.target_interval = target_interval

    def apply_transform(self, data):
        return clip_and_scale(data, self.clip_range, self.source_interval, self.target_interval)

    def __call__(self, sample):
        sample['image'] = self.apply_transform(sample['image'])
        return sample


class GaussianAdditiveNoise(object):
    """Add Gaussian noise to the input image.

    Examples
    --------
    The following transform could be used to add Gaussian additive noise with 20 HU to the image, and subsequently clip
    to [-300, 100]HU and rescale this to [0, 1].
    
    >>> transform = Compose([GaussianAdditiveNoise(0, 20), ClipAndScale([-300, 100], [-300, 100], [0, 1])])
    """
    def __init__(self, mean, stddev):
        """
        Adds Gaussian additive noise to the input image.

        Parameters
        ----------
        mean : float
        stddev : float
        """
        self.mean = mean
        self.stddev = stddev

    def apply_transform(self, data):
        return data + np.random.normal(loc=self.mean, scale=self.stddev, size=data.shape)

    def __call__(self, sample):
        sample['image'] = self.apply_transform(sample['image'])
        return sample
