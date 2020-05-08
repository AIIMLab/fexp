def elastic_deformation(image, vectorfield, int_time, int_method="euler", mode="bilinear", grid=()):
    """
    Apply elastic deformation to image by following the flow for small time of a given vectorfield.
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
