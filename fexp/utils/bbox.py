# coding=utf-8
import numpy as np

from typing import Optional, Union, List, Tuple


class BoundingBox:
    """BoundingBox class"""

    def __init__(
        self,
        data: Union[List, Tuple, np.ndarray],
        dtype: Optional[Union[np.dtype, str]] = None,
    ):
        data = np.asarray(data)
        if dtype:
            data = data.astype(dtype)

        self.coordinates, self.size = _split_bbox(data)
        self.data = data
        self.ndim = len(self.data) // 2

    @property
    def center(self):
        return self.coordinates + self.size / 2

    @property
    def dtype(self):
        return self.data.dtype

    def astype(self, dtype):
        return BoundingBox(self.data, dtype=dtype)

    def around_center(self, output_size):
        output_size = np.asarray(output_size)
        return BoundingBox(
            np.rint(_combine_bbox(self.center - output_size / 2, output_size))
        )

    def squeeze(self, axis=0):
        # TODO: Doesn't work for axis > 0
        """Add an extra axis to the bounding box."""
        bbox = self.data[:]
        coordinates, size = _split_bbox(bbox)
        coordinates = np.insert(coordinates, 0, 0, axis=axis)
        size = np.insert(size, 0, 1, axis=axis)
        bbox = _combine_bbox(coordinates, size)

        return BoundingBox(bbox)

    def crop_to_shape(self, shape):
        """
        Crop BoundingBox to the given shape, e.g. to crop the box to the image shape.

        Parameters
        ----------
        shape : list

        Returns
        -------
        BoundingBox
        """
        if len(shape) != self.ndim:
            raise ValueError(
                f"Shape has to have the same dimension as bounding box. Got {shape} and {self.ndim}."
            )

        # Crop to image shape
        new_coordinates = np.clip(self.coordinates, 0, shape)
        # Compute maximal size
        offset = shape - (self.size + new_coordinates)
        # Only keep negative values
        offset = np.clip(offset, None, 0)
        new_size = self.size + offset

        return BoundingBox(_combine_bbox(new_coordinates, new_size))

    def expand(self, shape):
        """
        Expand bounding box with given shape in all directions.

        Parameters
        ----------
        shape : list

        Returns
        -------
        BoundingBox
        """
        if len(shape) != self.ndim:
            raise ValueError(
                f"Shape has to have the same dimension as bounding box. Got {shape} and {self.ndim}."
            )
        shape = np.asarray(shape)
        new_coordinates = self.coordinates - shape
        new_size = self.size + shape
        return BoundingBox(_combine_bbox(new_coordinates, new_size))

    def relative_to(self, bbox):
        """
        Create the BoundingBox which is relative to the coordinate system of array.

        Parameters
        ----------
        bbox : BoundingBox

        Returns
        -------

        BoundingBox
        """
        coordinates_2, _ = bbox.coordinates, bbox.size
        new_coordinates = self.coordinates - coordinates_2
        return BoundingBox(_combine_bbox(new_coordinates, self.size))

    def to_mask(self, shape):
        """
        Convert bounding box to given shape. Box will be clipped prior to creating a mask.

        Parameters
        ----------
        shape : tuple

        Returns
        -------
        np.ndarray
        """
        new_box = self.crop_to_shape(shape)
        region_idx = tuple(
            slice(i, j)
            for i, j in zip(new_box.coordinates, new_box.coordinates + new_box.size)
        )

        mask = np.zeros(shape, dtype=bool)
        mask[region_idx] = True
        return mask

    def __add__(self, bbox):
        """Add operation:

        - Adding two bounding boxes returns the encapsulating BoundingBox.
        """
        # TODO: Asserts
        # assert_bbox(bbox)

        coordinates_2, size_2 = bbox.coordinates, bbox.size

        if self.ndim != bbox.ndim:
            raise ValueError(
                f"BoundingBoxes could not added together with dimensions {self.ndim} {bbox.ndim}."
            )

        # The encapsulating box starts at the minimal coordinates
        new_coordinates = np.stack([self.coordinates, coordinates_2]).min(axis=0)

        # The size is the maximum of all sizes
        new_size = np.abs(coordinates_2 - self.coordinates) + np.stack(
            [self.size, size_2]
        ).max(axis=0)
        new_size = np.stack([self.coordinates, coordinates_2, new_size]).max(axis=0)

        return BoundingBox(_combine_bbox(new_coordinates, new_size))

    def shift(self, x):
        """
        Shift the bounding box by a given vector.

        Parameters
        ----------
        vector : List or np.ndarray

        Returns
        -------

        BoundingBox
        """
        if len(x) not in [self.ndim, 1]:
            raise ValueError(
                f"Can only add a vector of dimension 1 or same dimension as BoundingBox Got {len(x)}."
            )

        x = np.asarray(x) + np.zeros_like(self.coordinates)  # Broadcast x to same shape

        new_coordinates = self.coordinates + np.asarray(x)

        return BoundingBox(_combine_bbox(new_coordinates, self.size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data.tolist())

    def __repr__(self):
        return f"BoundingBox({self.data}, ndim={self.ndim}, dtype={self.dtype}))"


def _split_bbox(bbox):
    """Split array into coordinates and size

    Parameters
    ----------
    bbox : tuple or ndarray. Given dimension n, first n coordinates are the starting point, the other n the size.

    Returns
    -------
    list of ndarrays
    """
    len_bbox = len(bbox)
    if not len_bbox % 2 == 0:
        raise ValueError(
            f"{bbox} needs to have a have a length which is divisible by 2."
        )
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
    # TODO: Check for integer dtype
    """Extract array from images, coordinates can be negative.

    Parameters
    ----------
    image : ndarray
       nD array
    bbox : list or tuple or BoundingBox
       array of the form (coordinates, size),
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

    region_idx = [
        slice(i, j)
        for i, j in zip(bbox_coords + l_offset, bbox_coords + bbox_size - r_offset)
    ]

    out = image[
        tuple(region_idx)
    ].copy()  # It can happen that this is a view, copying prevents this.

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    patch = pad_value * np.ones(bbox_size, dtype=image.dtype)
    patch_idx = [slice(i, j) for i, j in zip(l_offset, bbox_size - r_offset)]

    patch[tuple(patch_idx)] = out

    return patch


def assert_bbox(x: object):
    if not isinstance(x, BoundingBox):
        raise ValueError(f"Expected BoundingBox. Got {type(x)}.")
