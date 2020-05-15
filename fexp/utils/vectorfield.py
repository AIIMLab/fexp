# coding=utf-8
"""
Copyright (c) Fexp Contributors

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Module for constructing (random) vectorfields

import torch
from fexp.utils import convolve
import torch.nn.functional as nnf


class RandomGaussianVectorfield(object):
    """
    Construct random vectorfield by sampling vector components from Gaussian distribution
    """
    def __init__(self, grid_field, grid_target, mean_field, sigma_field, kernel=(), sigma_mollifier=(),
                 interp_mode="bilinear", conv_method="toeplitz"):
        """
        Parameters
        ----------
        grid_field: 2-tuple of ints
            specifies the grid on which the initial random vectorfield is sampled
        grid_target: 2-tuple of ints
            specifies the grid to which the random vectorfield is to be extended by interpolation
        mean_field: 1d float-valued Pytorch tensor with two components, optional
            mean value Gaussian used for sampling components vectorfield ([mean_field_x mean_field_y])
        sigma_field: 1d float-valued Pytorch tensor with two components, optional
            standard deviation Gaussian used for sampling components vectorfield ([sigma_field_x sigma_field_y])
        kernel: 2-tuple of 1d Pytorch tensors, optional
            separable kernel
        sigma_mollifier: 1d float-valued Pytorch tensor with two components (needed if no kernel is provided)
            level of smoothing (standard deviation Gaussian) in each spatial direction ([sigma_x sigma_y])
        interp_mode: str in {bilinear, constant}, optional
            interpolation technique used for upsampling and evaluating vectorfield outside grid
        conv_method: str in {toeplitz, torch, fft}, optional
            convolution method
        """

        # Initialization
        self.grid_field = grid_field
        self.grid_target = grid_target
        self.dim_spat = len(grid_field)
        self.mean_field = mean_field.view(self.dim_spat, 1, 1)
        self.sigma_field = sigma_field.view(self.dim_spat, 1, 1)

        if len(kernel) == 0:
            if sigma_mollifier:
                self.kernel = [convolve.gaussian_kernel(sigma) for sigma in sigma_mollifier]
                self.sigma_mollifier = sigma_mollifier
            else:
                raise ValueError("Mollification parameters should be specified if no kernel is provided")
        else:
            self.kernel = kernel

        self.interp_mode = interp_mode
        self.conv_method = conv_method

    def __call__(self):
        """
        Construct random vectorfield by sampling vector components from Gaussian distribution

        Returns
        -------
        3d float-valued Pytorch tensor of size [heigth width dim_spat]
            randomly sampled (smoothed) vectorfield
        """

        # Construct random vector field and mollify
        # Note: spatial components following standard geometrical ordering
        vectorfield = self.mean_field + self.sigma_field * torch.randn(self.dim_spat, *self.grid_field)
        vectorfield.unsqueeze_(0)
        for comp in range(self.dim_spat):
            vectorfield[:, comp] = convolve.conv2d_separable(vectorfield[:, comp], self.kernel, mode="trunc",
                                                             method=self.conv_method)

        # Interpolate vectorfield on grid on which image is sampled
        vectorfield = nnf.interpolate(vectorfield, size=self.grid_target, mode=self.interp_mode)
        return vectorfield.view(self.dim_spat, *self.grid_target).permute([1, 2, 0])
