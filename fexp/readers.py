"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import pathlib
import SimpleITK as sitk
import numpy as np


_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def read_image_as_sitk_image(filename):
    """
    Read file as a SimpleITK image trying to parse the error.

    Parameters
    ----------
    filename : pathlib.Path or str

    Returns
    -------
    SimpleITK image.
    """
    try:
        sitk_image = sitk.ReadImage(str(filename))
    except RuntimeError as error:
        if 'itk::ERROR' in str(error):
            error = str(error).split('itk::ERROR')[-1]

        raise RuntimeError(error)

    return sitk_image


def read_image(filename, dtype=None, no_metadata=False, force_2d=False):
    """Read medical image

    Parameters
    ----------
    filename : Path, str
        Path to image, can be any SimpleITK supported filename
    dtype : dtype
        The requested dtype the output should be cast.
    no_metadata : bool
        Do not output metadata
    force_2d : bool
        If this is set to true, first slice in first axis will be taken, if the size[0] == 1.

    Returns
    -------
    Image as ndarray and dictionary with metadata.
    """
    filename = pathlib.Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f'{filename} does not exist.')

    new_spacing = kwargs.get('spacing', False)
    if new_spacing and np.all(np.asarray(new_spacing) <= 0):
        new_spacing = False

    metadata = {}
    sitk_image = read_image_as_sitk_image(filename)

    # TODO: A more elaborate check for dicom can be needed, not necessarly all dicom files have .dcm as extension.
    if filename.suffix.lower() == '.dcm' and kwargs.get('dicom_keys', None):
        dicom_data = {}
        metadata_keys = sitk_image.GetMetaDataKeys()
        for v in kwargs['dicom_keys']:
            dicom_data[v] = None if v not in metadata_keys else sitk_image.GetMetaData(v).strip()
        metadata['dicom_tags'] = dicom_data

    orig_shape = sitk.GetArrayFromImage(sitk_image).shape
    if new_spacing:
        sitk_image, orig_spacing = resample_sitk_image(
            sitk_image,
            spacing=new_spacing,
            interpolator=kwargs.get('interpolator', None),
            fill_value=0
        )
        metadata.update(
            {'orig_spacing': tuple(orig_spacing), 'orig_shape': orig_shape})

    image = sitk.GetArrayFromImage(sitk_image)

    metadata.update({
        'filename': filename.resolve(),
        'depth': sitk_image.GetDepth(),
        'spacing': sitk_image.GetSpacing(),
        'origin': sitk_image.GetOrigin(),
        'direction': sitk_image.GetDirection()
    })

    if force_2d:
        if not image.shape[0] == 1:
            raise ValueError(f'Forcing to 2D while the first dimension is not 1.')
        image = image[0]

    if dtype:
        image = image.astype(dtype)

    if no_metadata:
        return image

    return image, metadata


def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.

    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int

    Returns
    -------
    SimpleITK image.
    """
    if isinstance(sitk_image, (str, pathlib.Path)):
        sitk_image = read_image_as_sitk_image(sitk_image)
    num_dim = sitk_image.GetDimension()
    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: #  8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing]*num_dim
    else:
        new_spacing = [float(s) if s else orig_spacing[idx] for idx, s in enumerate(spacing)]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    # SimpleITK expects lists
    new_size = [int(s) if spacing[idx] else int(orig_size[idx]) for idx, s in enumerate(new_size)]

    resample_filter = sitk.ResampleImageFilter()
    resampled_sitk_image = resample_filter.Execute(
        sitk_image,
        new_size,
        sitk.Transform(),
        sitk_interpolator,
        orig_origin,
        new_spacing,
        orig_direction,
        fill_value,
        orig_pixelid
    )

    return resampled_sitk_image, orig_spacing
