import tensorflow as tf

from attack.utils import get_gaussian_kernel
import numpy as np


def get_ssim_value_by_tensor(old_image_gray, new_image_gray):
    """
    compute ssim value between old_image and new_image
    :param old_image_gray: tensor np.array
    :param new_image_gray: tensor np.array
    :return: tensor float
    """
    old_image_gray = (old_image_gray / 2 + 0.5) * 255
    new_image_gray = (new_image_gray / 2 + 0.5) * 255
    gaussian_kernel_1d = get_gaussian_kernel()
    old_image_gray_squared = old_image_gray ** 2
    new_image_gray_squared = new_image_gray ** 2
    old_image_gray_mu = old_image_gray
    new_image_gray_mu = new_image_gray
    # old_image_gray_mu = convolve_gaussian_2d(
    #     old_image_gray, gaussian_kernel_1d)
    # new_image_gray_mu = convolve_gaussian_2d(
    #     new_image_gray, gaussian_kernel_1d)
    old_image_gray_mu_squared = old_image_gray_mu ** 2
    new_image_gray_mu_squared = new_image_gray_mu ** 2
    old_image_gray_sigma_squared = old_image_gray_squared
    new_image_gray_sigma_squared = new_image_gray_squared
    # old_image_gray_sigma_squared = convolve_gaussian_2d(
    #     old_image_gray_squared, gaussian_kernel_1d)
    # new_image_gray_sigma_squared = convolve_gaussian_2d(
    #     new_image_gray_squared, gaussian_kernel_1d)
    old_image_gray_sigma_squared -= old_image_gray_mu_squared
    new_image_gray_sigma_squared -= new_image_gray_mu_squared

    image_mat = old_image_gray * new_image_gray
    image_mat_sigma = image_mat
    # image_mat_sigma = convolve_gaussian_2d(
    #     image_mat, gaussian_kernel_1d)
    img_mat_mu = old_image_gray_mu * new_image_gray_mu
    img_mat_sigma = image_mat_sigma - img_mat_mu

    c_1 = (0.01 * 255) ** 2
    c_2 = (0.03 * 255) ** 2
    num_ssim = ((2 * img_mat_mu + c_1) *
                (2 * img_mat_sigma + c_2))
    den_ssim = (
            (old_image_gray_mu_squared + new_image_gray_mu_squared +
             c_1) *
            (old_image_gray_sigma_squared +
             new_image_gray_sigma_squared + c_2))
    ssim_map = num_ssim / den_ssim
    index = tf.reduce_mean(ssim_map)
    old_image_gray = (old_image_gray / 255.0 - 0.5) * 2
    new_image_gray = (new_image_gray / 255.0 - 0.5) * 2
    return index


def get_ssim_value(old_image_gray, new_image_gray):
    """
    compute ssim value between old_image and new_image
    :param old_image_gray:  np.array
    :param new_image_gray:  np.array
    :return: float
    """
    old_image_gray = (old_image_gray / 2 + 0.5) * 255
    new_image_gray = (new_image_gray / 2 + 0.5) * 255
    gaussian_kernel_1d = get_gaussian_kernel()
    old_image_gray_squared = old_image_gray ** 2
    new_image_gray_squared = new_image_gray ** 2
    old_image_gray_mu = old_image_gray
    new_image_gray_mu = new_image_gray
    # old_image_gray_mu = convolve_gaussian_2d(
    #     old_image_gray, gaussian_kernel_1d)
    # new_image_gray_mu = convolve_gaussian_2d(
    #     new_image_gray, gaussian_kernel_1d)
    old_image_gray_mu_squared = old_image_gray_mu ** 2
    new_image_gray_mu_squared = new_image_gray_mu ** 2
    old_image_gray_sigma_squared = old_image_gray_squared
    new_image_gray_sigma_squared = new_image_gray_squared
    # old_image_gray_sigma_squared = convolve_gaussian_2d(
    #     old_image_gray_squared, gaussian_kernel_1d)
    # new_image_gray_sigma_squared = convolve_gaussian_2d(
    #     new_image_gray_squared, gaussian_kernel_1d)
    old_image_gray_sigma_squared -= old_image_gray_mu_squared
    new_image_gray_sigma_squared -= new_image_gray_mu_squared

    image_mat = old_image_gray * new_image_gray
    image_mat_sigma = image_mat
    # image_mat_sigma = convolve_gaussian_2d(
    #     image_mat, gaussian_kernel_1d)
    img_mat_mu = old_image_gray_mu * new_image_gray_mu
    img_mat_sigma = image_mat_sigma - img_mat_mu

    c_1 = (0.01 * 255) ** 2
    c_2 = (0.03 * 255) ** 2
    num_ssim = ((2 * img_mat_mu + c_1) *
                (2 * img_mat_sigma + c_2))
    den_ssim = (
            (old_image_gray_mu_squared + new_image_gray_mu_squared +
             c_1) *
            (old_image_gray_sigma_squared +
             new_image_gray_sigma_squared + c_2))
    ssim_map = num_ssim / den_ssim
    index = np.mean(ssim_map)
    old_image_gray = (old_image_gray / 255.0 - 0.5) * 2
    new_image_gray = (new_image_gray / 255.0 - 0.5) * 2
    return index
