"""
Copyright (c) Nikita Moriakov, Jonas Teuwen and Ray Sheombarsing

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np


class BoundingBox(object):
    """BoundingBox class
    """
    def __init__(self, bbox, dtype=np.int):
        if isinstance(bbox, BoundingBox):
            self.bbox = bbox.bbox
        elif isinstance(bbox, (list, tuple, np.ndarray)):
            self.bbox = np.asarray(bbox)
        else:
            raise ValueError(f'BoundingBox only accepts BoundingBox, list, tuple or np.ndarrays as input.')

        self.dtype = dtype
        if dtype:
            self.bbox = self.bbox.astype(dtype)

        self.coordinates, self.size = _split_bbox(bbox)
        self.ndim = len(self.bbox) // 2

    @property
    def center(self):
        return self.coordinates + self.size / 2

    def bounding_box_around_center(self, output_size):
        output_size = np.asarray(output_size)
        return BoundingBox(np.rint(_combine_bbox(self.center - output_size / 2, output_size)))

    def squeeze(self, axis=0):
        """Add an extra axis to the bounding box."""
        bbox = self.bbox[:]
        coordinates, size = _split_bbox(bbox)
        coordinates = np.insert(coordinates, 0, 0, axis=axis)
        size = np.insert(size, 0, 1, axis=axis)
        bbox = _combine_bbox(coordinates, size)

        return BoundingBox(bbox)

    def astype(self, dtype):
        return BoundingBox(self.bbox, dtype=dtype)

    def __add__(self, x):
        """Add operation:

        - Adding two bounding boxes returns the encapsulating BoundingBox.
        - Adding a vector shifts the center of the BoundingBox.
        """
        if isinstance(x, BoundingBox):
            self.coordinates_2, self.size_2 = x.coordinates, x.size

            if self.ndim == len(x.ndim):
                raise ValueError(f'ValueError: '
                                 f'BoundingBoxes could not added together with dimensions {self.ndim} {x.ndim}.')

            # The encapsulating box starts at the minimal coordinates
            new_coordinates = np.stack([self.coordinates, self.coordinates_2]).min(axis=0)

            # The size is the maximum of all sizes
            new_size = np.abs(self.coordinates_2 - self.coordinates) + np.stack([self.size, self.size_2]).max(axis=0)
            new_size = np.stack([self.coordinates, self.coordinates_2, new_size]).max(axis=0)

            return BoundingBox(_combine_bbox(new_coordinates, new_size))

        else:
            x = np.asarray(x) + np.zeros_like(self.coordinates)  # Broadcast x to same shape

            if len(x) is not self.ndim:
                raise ValueError(f'ValueError: Can only add a vector of same dimension as BoundingBox Got {len(x)}.')
            new_center = self.center + np.asarray(x)
            new_coordinates = new_center - self.size / 2

            return BoundingBox(_combine_bbox(new_coordinates, self.size), dtype=self.dtype)

    def __len__(self, x):
        return len(self.bbox)

    def __getitem__(self, idx):
        return self.bbox[idx]

    def __iter__(self):
        return iter(self.bbox)

    def __repr__(self):
        return f'BoundingBox({self.bbox}))'


def _split_bbox(bbox):
    """Split bbox into coordinates and size

    Parameters
    ----------
    bbox : tuple or ndarray. Given dimension n, first n coordinates are the starting point, the other n the size.

    Returns
    -------
    list of ndarrays
    """
    len_bbox = len(bbox)
    if not len_bbox % 2 == 0:
        raise ValueError(f'{bbox} needs to have a have a length which is divisible by 2.')
    ndim = len_bbox // 2
    bbox_coords = bbox[:ndim]
    bbox_size = bbox[ndim:]
    return np.asarray(bbox_coords), np.asarray(bbox_size)


def _combine_bbox(bbox_coords, bbox_size):
    """Combine coordinates and size into a bounding box.

    Parameters
    ----------
    bbox_coords : tuple or ndarray
    bbox_size : tuple or ndarray

    Returns
    -------
    tuple

    """
    bbox_coords = np.asarray(bbox_coords)
    bbox_size = np.asarray(bbox_size)
    bbox = tuple(bbox_coords.tolist() + bbox_size.tolist())
    return bbox


def bounding_box(mask):
    """
    Computes the bounding box of a mask
    Parameters
    ----------
    mask : array-like
        Input mask

    Returns
    -------
    BoundingBox
    """
    bbox_coords = []
    bbox_sizes = []
    for idx in range(mask.ndim):
        axis = tuple([i for i in range(mask.ndim) if i != idx])
        nonzeros = np.any(mask, axis=axis)
        min_val, max_val = np.where(nonzeros)[0][[0, -1]]
        bbox_coords.append(min_val)
        bbox_sizes.append(max_val - min_val + 1)

    return BoundingBox(_combine_bbox(bbox_coords, bbox_sizes))


def crop_to_bbox(image, bbox, pad_value=0):
    """Extract bbox from images, coordinates can be negative.

    Parameters
    ----------
    image : ndarray
       nD array
    bbox : list or tuple or BoundingBox
       bbox of the form (coordinates, size),
       for instance (4, 4, 2, 1) is a patch starting at row 4, col 4 with height 2 and width 1.
    pad_value : number
       if bounding box would be out of the image, this is value the patch will be padded with.

    Returns
    -------
    ndarray
        Numpy array of data cropped to BoundingBox
    """
    if not isinstance(bbox, BoundingBox):
        bbox = BoundingBox(bbox)
    # Coordinates, size
    bbox_coords, bbox_size = bbox.coordinates, bbox.size

    # Offsets
    l_offset = -bbox_coords.copy()
    l_offset[l_offset < 0] = 0

    r_offset = (bbox_coords + bbox_size) - np.array(image.shape)
    r_offset[r_offset < 0] = 0

    region_idx = [slice(i, j) for i, j
                  in zip(bbox_coords + l_offset,
                         bbox_coords + bbox_size - r_offset)]

    out = image[tuple(region_idx)].copy()  # It can happen that this is a view, copying prevents this.

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    patch = pad_value * np.ones(bbox_size, dtype=image.dtype)
    patch_idx = [slice(i, j) for i, j
                 in zip(l_offset, bbox_size - r_offset)]

    patch[tuple(patch_idx)] = out

    return patch
