# encoding: utf-8
__author__ = 'Ray Sheombarsing'

# Module for applying (deterministic) transformations to (2d) images.

import torch
import torch.nn.functional as nnf
from augmentation import convolution


def compute_grid(discret_size):
    """
    Compute 2d grid of the form {(j, i) : 0 <= j <= n_y - 1, 0 <= i <= n_x - 1} stored in a n_y x n_x x 2 matrix

    Parameters
    ----------
    discret_size: 2-tuple of ints
        specifies discretization size in each direction

    Returns
    -------
    float-valued Pytorch-tensor of size [n_y n_x 2]
        grid associated to prescribed discretization sizes
    """
    mesh = torch.meshgrid((torch.arange(discret_size[0]), torch.arange(discret_size[1])))
    return torch.stack(mesh, 2).flip([2]).float()


def rescale(pts, domain, target):
    """
    Apply linear transformation from domain to target along last dimension
    todo: ensure transformation is aligned with the (prescribed) dimension of interest!

    Parameters
    ----------
    pts: Pytorch tensor of size [... d]
        input to be transformed
    domain: Pytorch tensor of size (d, 2)
        domain of linear map to be applied to input
    target: Pytorch tensor of size (d, 2)
        target to which domain is to be mapped

    Returns
    -------
    float-valued Pytorch tensor
        points rescaled according to prescribed linear transformation
    """
    target = target.float()
    domain = domain.float()
    return (target[:, 1] - target[:, 0]) / (domain[:, 1] - domain[:, 0]) * (pts - domain[:, 0]) + target[:, 0]


def affine_transformation(image, lin_transform, shift=torch.zeros(2, 1), origin=None, mode='bilinear', grid=()):
    """
    Apply affine transformation to image

    Parameters
    ----------
    image: 3d float-valued Pytorch tensor [num_images height width]
        images to be transformed
    lin_transform: 2d float-valued Pytorch tensor of size 2 x 2
        matrix which prescribes the linear part of the transformation
    shift: 2d float-valued Pytorch tensor of size 2 x 1
        shift associated to affine transformation ([shift_x shift_y])
    origin: 2d float-valued Pytorch tensor of size 2 x 1
        origin associated to coordinates affine transformation ([origin_x origin_y])
    mode: str in {bilinear, constant}
        interpolation scheme
    grid: float-valued Pytorch-tensor of size [n_y n_x 2], optional
        gridpoints {(j, i) : 0 <= j <= n_y - 1, 0 <= i <= n_x - 1}

    Returns
    -------
    image_transformed: 3d float-valued Pytorch tensor [num_images height width]
        transformed images
    """

    # Initialization
    dim_spat = 2
    image_shape = image.size()
    discret_size = image_shape[len(image_shape) - dim_spat::]

    # Geometric quantities
    shift = shift.view(1, 1, dim_spat)
    identity = torch.tensor([[1, 0], [0, 1]]).float()
    if len(grid) == 0:
        grid = compute_grid(discret_size)

    # Apply affine transformation (with prescribed origin)
    if torch.all(lin_transform.eq(identity)):
        transformed_grid = grid - shift
    else:
        if not origin:
            origin = torch.tensor([(discret_size[1] - 1) / 2, (discret_size[0] - 1) / 2]).view(1, 1, dim_spat)
        transformed_grid = torch.matmul(lin_transform, (grid - (shift + origin)).view(*discret_size, dim_spat, 1))
        transformed_grid = transformed_grid.view(*discret_size, dim_spat) + origin

    # Take "flipped" dimension in y-direction into account, rescale to [-1, 1] and interpolate
    transformed_grid[:, :, 1] = discret_size[0] - 1 - transformed_grid[:, :, 1]
    transformed_grid = rescale(transformed_grid, torch.tensor([[0, discret_size[1] - 1], [0, discret_size[0] - 1]]),
                               torch.tensor([[-1, 1], [-1, 1]]))
    transformed_image = nnf.grid_sample(image.view(1, *image_shape), transformed_grid.unsqueeze(0), mode=mode)

    # Flip image back to usual "upside-down" convention
    return transformed_image[0, :, torch.arange(discret_size[0] - 1, -1, -1), :]


def flip(image, axis):
    """
    Flip 2d image along prescribed axis with given probability. Implementation is specific to 2d setting to avoid
    permutation of axes.

    Parameters
    ----------
    image :3d float-valued Pytorch tensor [num_images height width]
        images to be flipped
    axis: str in {horizontal, vertical}
        index of axis to flip

    Returns
    -------
    3d float-valued Pytorch tensor [num_images height width]
        flipped images
    """
    if axis == "horizontal":
        return image[:, :, torch.arange(image.size()[2] - 1, -1, -1)]
    elif axis == "vertical":
        return image[:, torch.arange(image.size()[1] - 1, -1, -1), :]


def gaussian_blur(image, kernel=(), sigma=(), conv_method='toeplitz'):
    """
    Blur 2d image by convolving with Gaussian kernel

    Parameters
    ----------
    image: 3d float-valued Pytorch tensor [num_images height width]
        images to be blurred
    kernel: 2-tuple of 1d Pytorch tensors, optional
        separable kernel
    sigma: 1d float-valued Pytorch tensor with two components (needed if no kernel is provided)
        standard deviation of Gaussian in each spatial direction (sigma = [sigma_x sigma_y])
    conv_method: str in {toeplitz, torch, fft}, optional
        convolution method

    Returns
    -------
    3d float-valued Pytorch tensor [num_images heigth width]
        blurred image
    """
    if len(kernel) == 0:
        kernel = [convolution.gaussian_kernel(sigma_comp) for sigma_comp in sigma]
    return convolution.conv2d_separable(image, kernel, mode='trunc', method=conv_method)


def elastic_deformation(image, vectorfield, int_time, int_method="euler", mode="bilinear", grid=()):
    """
    Apply elastic deformation to image by following the flow of a given vectorfield
    Todo: code contains unnecessary lines specific to the 2d case; this can be easily generalized to
    arbitrary dimensions.

    Parameters
    ----------
    image: 3d float-valued Pytorch tensor [num_images height width]
        images to be deformed
    vectorfield: 3d float-valued Pytorch tensor of size [heigth width dim_spat]
        (Lipschitz) continuous vectorfield
    int_time: float
        integration time
    mode: str in {bilinear, constant}, optional
        interpolation technique used for evaluating image outside grid
    grid: float-valued Pytorch-tensor of size [heigth width dim_spat], optional
        gridpoints {(j, i) : 0 <= j <= n_y - 1, 0 <= i <= n_x - 1}

    Returns
    -------
    3d float-valued Pytorch tensor [num_images heigth width]
        deformed image
    """

    # Initialization: dimensions
    dim_spat = 2
    image_shape = image.size()
    discret_size = image_shape[len(image_shape) - dim_spat::]

    if len(grid) == 0:
        grid = compute_grid(discret_size)

    # Flow backwards (Euler). Note: vectorfield[j, i, :] follows standard geometrical ordering
    if int_method == "euler":
        deformed_grid = grid - int_time * vectorfield
    elif int_method == "rk4":
        raise NotImplementedError("Runga Kutta is not yet implemented")

    # Rescale grid for interpolation
    deformed_grid = rescale(deformed_grid, torch.tensor([[0, discret_size[1] - 1], [0, discret_size[0] - 1]]),
                            torch.tensor([[-1, 1], [-1, 1]]))

    # Interpolate image
    return nnf.grid_sample(image.unsqueeze(0), deformed_grid.unsqueeze_(0), mode=mode).view(*image_shape)
