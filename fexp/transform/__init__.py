# coding=utf-8
"""
Copyright (c) Fexp Contributors

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import random
import numpy as np
from fexp.transform.affine import rescale, affine


class Identity(object):
    """Identity transform (i.e. leave the input unchanged). Can be convenient when random sampling between different
    augmentations.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class RandomTransform(object):
    """Select a transform randomly from a list"""
    def __init__(self, transforms, choose_weight=None):
        """
        Given a weight, a transform is chosen from a list.

        Parameters
        ----------
        transforms : list
        choose_weight : list or np.ndarray
        """
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        self.choose_weight = choose_weight

        self._num_transforms = len(transforms)

    def __call__(self, sample):
        if self.choose_weight:
            idx = random.choices(range(self._num_transforms), self.choose_weight)[0]
        else:
            idx = np.random.randint(0, self._num_transforms)
        transform = self.transforms[idx]
        sample = transform(sample)  # pylint: disable=not-callable
        return sample

    def __repr__(self):
        repr_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            repr_string += '\n'
            repr_string += f'    {transform}'
        repr_string += '\n)'
        return repr_string


class Compose(object):
    """Compose several transformations together, for instance ClipAndScale and a flip.

    Code based on torchvision: https://github.com/pytorch/vision, but got forked from there as torchvision has some
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

    def __repr__(self):
        repr_string = f'{self.__class__.__name__}(clip_range={self.clip_range}, ' \
                      f'source_interval={self.source_interval}, ' \
                      f'target_interval={self.target_interval})'
        return repr_string
