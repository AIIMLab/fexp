# encoding: utf-8
__author__ = 'Ray Sheombarsing'

# Module for constructing random transformations which can be applied to (2d) images.

import torch
import abc
import random
from augmentation import transforms
from augmentation import convolution
import torch.nn.functional as nnf
import matplotlib.pyplot as plt
from matplotlib import rc


class RandomParamInit(object):
    """
    Random parameter initializer
    """
    def __init__(self, range_params=(), init_rand_params=None):
        """
        Constructor: initialize parameter range and sampling method

        Parameters
        ----------
        range_params: float-valued Pytorch tensor of size num_params x 2
            each row corresponds to a prescribed range out which a sample is to be drawn
        init_rand_params: function mapping float-valued Pytorch tensor -> (num_params,) Pytorch tensor
            custom defined sampling method (uniform by default)
        """
        self.range_params = range_params
        if not init_rand_params:
            self.init_rand_params = lambda ran: (ran[:, 1] - ran[:, 0]) * torch.rand(ran.size()[0]) + ran[:, 0]
        else:
            self.init_rand_params = init_rand_params

    def __call__(self):
        if len(self.range_params) > 0:
            return self.init_rand_params(self.range_params)
        else:
            return self.init_rand_params()


class RandomTransform(abc.ABC):
    """
    2d transformation of (image, mask) pair
    """
    def __init__(self, rand_param_init=None, discret_size=None):
        """
        Constructor: initialize transforms and grid for randomly transforming images and masks

        Parameters
        ----------
        rand_param_init: RandomParamInit
            initialization of random parameters (e.g. an angle, direction, etc.)
        discret_size: 2-tuple of ints, optional
            specifies discretization size in each direction
        """
        self.discret_size = discret_size
        if discret_size:
            self.grid = transforms.compute_grid(discret_size)
        else:
            self.grid = None
        self.rand_param_init = rand_param_init

    @abc.abstractmethod
    def transform(self, image, mask, params, grid=None):
        """
        Applies a 2d transform to a collection of (image, mask) pairs

        Parameters
        ----------
        image: 3d float-valued Pytorch tensor [num_images height width]
            images to be transformed
        mask: 3d float-valued Pytorch tensor [num_classes height width]
            masks to be transformed
        params: float-valued Pytorch tensor of size (num_params, )
            parameters needed to perform transform (e.g. an angle, axis, etc.)
        grid: float-valued Pytorch-tensor of size [n_x n_y 2], optional
            gridpoints {(j,i) : 0 <= j <= n_y - 1, 0 <= x <= n_x - 1}

        Returns
        -------
        2-tuple of 3d float-valued Pytorch tensor [num_images height width]
            transformed (image, mask) pair
        """
        pass

    def __call__(self, sample):
        """
        Apply prescribed transformations to slices of image and channels (classes) of mask

        Parameters
        ----------
        sample: dictionary with entries 'image' and 'mask' containing tensors [batch_size num_channels heigth width]
            sample (image, mask) returned by a dataloader associated to SegData

        Returns
        -------
        sample: dictionary with entries 'image' and 'mask'
            transformed image and mask (in place)
        """

        # Initialization
        shape_image = sample['image'].size()
        shape_mask = sample['mask'].size()

        # Initialize random parameters
        if self.rand_param_init:
            params = self.rand_param_init()
        else:
            params = None

        # Apply transform
        image, mask = self.transform(sample['image'].view(shape_image[0] * shape_image[1], *shape_image[2::]),
                                     sample['mask'].view(shape_mask[0] * shape_mask[1], *shape_mask[2::]),
                                     params, grid=self.grid)

        # Reshape image and mask to original "batched" shape
        sample['image'] = image.view(*shape_image)
        sample['mask'] = mask.view(*shape_mask)

        return sample


class DataAug:
    """
    Class which represents a collection of random transformations
    """

    def __init__(self, transforms):
        """
        Initialize list of transforms which can be applied to (image, mask) pair.
        Todo: transforms should consist of a list of basis transformations which are automatically composed

        Parameters
        ----------
        transforms: list[RandomTransforms]
            a list containing random transforms
        """
        self.transforms = transforms

    def __call__(self, sample):
        """
        Apply a random transform to a sample (image, mask) pair

        Parameters
        ----------
        sample: dictionary with entries 'image' and 'mask' containing tensors [batch_size num_channels heigth width]
            sample (image, mask) returned by a dataloader associated to SegData

        Returns
        -------
        sample: dictionary with entries 'image' and 'mask'
            transformed image and mask (in place)
        """
        return random.sample(self.transforms, 1)[0](sample)


class Identity():
    """
    Identity transformation
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        """
        Apply identity transformation to sample

        Parameters
        ----------
        sample: dictionary with entries 'image' and 'mask' containing tensors [batch_size num_channels heigth width]
            sample (image, mask) returned by a dataloader associated to SegData

        Returns
        -------
        sample: dictionary with entries 'image' and 'mask'
            same image and mask pair (in place)
        """
        return sample


class RandomRotate(RandomTransform):
    """
    Random 2d rotation of (image, mask) pair
    """
    def __init__(self, range_angle, discret_size):
        """
        Constructor: initialize dimensions and range for rotations

        Parameters
        ----------
        discret_size: 2-tuple of ints
            dimension size of image [height width]
        range_angle: float-valued Pytorch tensor of size 1 x 2
            interval from which angles are sampled
        """
        super().__init__(RandomParamInit(range_params=range_angle), discret_size=discret_size)

    @staticmethod
    def inv_rot_mat(angle):
        """
        Construct inverse 2d rotation matrix

        Parameters
        ----------
        angle: float-valued 1d Pytorch tensor
            angle in radians

        Returns
        -------
        float-valued 2d Pytorch tensor of size 2 x 2
            2d inverse rotation matrix
        """
        return torch.tensor([[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]])

    def transform(self, image, mask, angle, grid):
        return transforms.affine_transformation(image, self.inv_rot_mat(angle), mode='bilinear', grid=self.grid), \
               transforms.affine_transformation(mask, self.inv_rot_mat(angle), mode='nearest', grid=self.grid)


class RandomShift(RandomTransform):
    """
    Random 2d shift of (image, mask) pair
    """
    def __init__(self, range_shift, discret_size):
        """
        Constructor: initialize dimensions and random shift

        Parameters
        ----------
        discret_size: 2-tuple of ints
            dimension size of image [height width]
        range_shift: float-valued Pytorch tensor of size 2 x 2
            interval from which components of the shift are sampled
        """
        super().__init__(RandomParamInit(range_params=range_shift), discret_size=discret_size)
        self.identity = torch.tensor([[1, 0], [0, 1]]).float()

    def transform(self, image, mask, direction, grid):
        return transforms.affine_transformation(image, self.identity, shift=direction, mode='bilinear', grid=self.grid), \
               transforms.affine_transformation(mask, self.identity, shift=direction, mode='nearest', grid=self.grid)


class RandomFlip(RandomTransform):
    """
    Random flip of 2d (image, mask) pair
    """
    def __init__(self, range_prob, axis='horizontal', tau=0.5):
        """
        Constructor: initialize random flip parameters

        Parameters
        ----------
        range_prob: float-valued Pytorch tensor of size 1 x 2
            range from which flip probability is sampled
        axis: str in {horizontal, vertical}
            index of axis to flip
        tau: float in (0, 1)
            image is flipped if prob > tau
        """
        super().__init__(RandomParamInit(range_params=range_prob))
        self.axis = axis
        self.tau = tau

    def flip(self, image, prob):
        """
        Flip image along prescribed axis if prob > tau

        Parameters
        ----------
        image: 3d float-valued Pytorch tensor [num_images height width]
            images to be flipped
        prob: float
            images are flipped if prob > tau

        Returns
        -------
        3d float-valued Pytorch tensor [num_images height width]
            flipped images
        """
        if prob > self.tau:
            return transforms.flip(image, self.axis)
        else:
            return image

    def transform(self, image, mask, prob, grid):
        return self.flip(image, prob), self.flip(mask, prob)


class RandomNoise(RandomTransform):
    """
    Add random noise to 2d (image, mask) pair
    """
    def __init__(self, type_noise, dist_params):
        """
        Constructor: initialize parameters noise

        Parameters
        ----------
        type_noise: str in {poisson, gaussian}
            specifies the type of noise
        dist_params: 1d float-valued Pytorch tensor
            the parameters associated to the probability distribution of the noise; [mean var] if Gaussian and
            [lambda] if poisson
        """
        super().__init__()
        self.type_noise = type_noise
        self.dist_params = dist_params
        if type_noise == "poisson":
            self.prob_dist = torch.distributions.poisson.Poisson(self.dist_params[0])
        elif type_noise == "gaussian":
            self.prob_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.dist_params[0].view(1, ),
                                                                                        self.dist_params[1].view(1, 1))

    def noise(self, image):
        """
        Add noise to image

        Parameters
        ----------
        image: 3d float-valued Pytorch tensor [num_images height width]
            images to which noise will be added

        Returns
        -------
        3d float-valued Pytorch tensor [num_images height width]
             images with noise
        """
        image_shape = image.size()
        sample_noise = self.prob_dist.sample(image_shape).view(*image_shape)
        if self.type_noise == "gaussian":
            return image + sample_noise
        elif self.type_noise == "poisson":
            raise NotImplementedError("Poisson noise is not yet implemented")

    def transform(self, image, mask, params, grid):
        return self.noise(image), mask


class GaussianBlur(RandomTransform):
    """
    Blur (image, mask) pair
    """
    def __init__(self, range_sigma):
        super().__init__(RandomParamInit(range_params=range_sigma))

    def transform(self, image, mask, sigma, grid):
        return transforms.gaussian_blur(image, sigma=sigma), mask


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
                self.kernel = [convolution.gaussian_kernel(sigma) for sigma in sigma_mollifier]
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
            vectorfield[:, comp] = convolution.conv2d_separable(vectorfield[:, comp], self.kernel, mode="trunc",
                                                                method=self.conv_method)

        # Interpolate vectorfield on grid on which image is sampled
        vectorfield = nnf.interpolate(vectorfield, size=self.grid_target, mode=self.interp_mode)
        return vectorfield.view(self.dim_spat, *self.grid_target).permute([1, 2, 0])

    def plot(self, vectorfield, path_out, name):
        """
        Depict a 2d randomly generated vectorfield

        Parameters
        ----------
        vectorfield: 3d float-valued Pytorch tensor of size [heigth width dim_spat]
            vectorfield to be depicted
        path_out: Path
            path where figure will be stored
        name : str
            filename of figure
        """

        # Set font and tex rendering
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
        rc('text', usetex=True)

        if not path_out.exists():
            path_out.mkdir()

        # Plot vectorfield
        fig, ax = plt.subplots()
        mesh_x, mesh_y = torch.meshgrid(torch.arange(self.grid_target[1]), torch.arange(self.grid_target[0]))
        ax.quiver(mesh_x, mesh_y, vectorfield[:, :, 0], vectorfield[:, :, 1], units='width')

        # Set axis labels and title
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        title = f'$\mu_{{field}} = {self.mean_field.flatten().tolist()}$, ' \
                + f'$\sigma_{{field}} = {self.sigma_field.flatten().tolist()}$, ' \
                + f'$\sigma_{{mollifier}} = {self.sigma_mollifier}$'
        ax.set_title(title)

        # Export figure
        fig.savefig((path_out / name).with_suffix(".pdf"), dpi=300, bbox_inches='tight')
        plt.close(fig)


class RandomDeformation(RandomTransform):
    """
    Deform (image, mask) pair using randomly generated (Gaussian) vectorfield
    """
    def __init__(self, discret_size_image, grid_field, mean_field, sigma_field, sigma_mollifier, int_time,
                 interp_mode_field="bilinear", interp_mode_image="bilinear", conv_method="toeplitz", grid=()):

        """
        Constructor: initialize parameters vectorfield and ODE integration for elastic deformation

        Parameters
        ----------
        discret_size_image: 2-tuple of ints
            specifies discretization size of image in each direction
        grid_field: 2-tuple of ints
            specifies the grid on which the initial random vectorfield is sampled
        mean_field: 1d float-valued Pytorch tensor with two components, optional
            mean value Gaussian used for sampling components vectorfield ([mean_field_x mean_field_y])
        sigma_field: 1d float-valued Pytorch tensor with two components, optional
            standard deviation Gaussian used for sampling components vectorfield ([sigma_field_x sigma_field_y])
        sigma_mollifier: 1d float-valued Pytorch tensor with two components
            level of smoothing (standard deviation Gaussian) in each spatial direction ([sigma_x sigma_y])
        int_time: float
            integration time
        interp_mode_field: str in {bilinear, constant}, optional
            interpolation technique used for upsampling and evaluating vectorfield outside grid
        interp_mode_image: str in {bilinear, constant}, optional
            interpolation technique used for evaluating image outside grid
        conv_method: str in {toeplitz, torch, fft}, optional
            convolution method
        """

        super().__init__(RandomParamInit(init_rand_params=RandomGaussianVectorfield(grid_field, discret_size_image,
                                                                                    mean_field, sigma_field,
                                                                                    sigma_mollifier=sigma_mollifier,
                                                                                    interp_mode=interp_mode_field,
                                                                                    conv_method=conv_method)),
                         discret_size=discret_size_image)

        self.int_time = int_time
        self.interp_mode_image = interp_mode_image

    def transform(self, image, mask, vectorfield, grid):
        return transforms.elastic_deformation(image, vectorfield, self.int_time, grid=self.grid,
                                              mode=self.interp_mode_image), \
               transforms.elastic_deformation(mask, vectorfield, self.int_time, grid=self.grid,
                                              mode='nearest')
