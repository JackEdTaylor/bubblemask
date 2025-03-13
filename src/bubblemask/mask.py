from PIL import Image
import numpy as np
from skimage.morphology import binary_dilation
import warnings
from . import build, apply

def bubbles_mask (im, mu_x=None, mu_y=None, sigma=np.array([5]), bg=0, **kwargs):
    """
    Apply the bubbles mask to a given PIL image.
    
    Parameters
    ----------
    im : PIL.Image.Image
        The image to apply the bubbles mask to.
    mu_x, mu_y : array_like, optional
        Indices of bubble locations in x (`mu_x`) and y (`mu_y`) dimensions. Can be floats. The default (`None`) sets to random locations.
    sigma : array_like
        Array of sigmas for the spread of the bubbles. `n` is inferred from the length of this array.
    bg : array_like
        Value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB, or 4 for RGBA.
    **kwargs :
        Additional arguments passed to `build.build_mask`, e.g., `scale` and `sum_merge`.

    Returns
    -------
    im_out : PIL.Image-Image
        Bubble-masked image.
    mask : np.array
        The numpy array used as a mask for `im`, with the same shape as `im`.
    mu_x, mu_y, sigma : np.array
        The locations of bubbles in x (`mu_x`) and y (`mu_y`) dimensions, with corresponding `sigma` values.
    """
    
    n = len(sigma)  # get n bubbles
    sh = np.asarray(im).shape  # get shape
    
    # generate distributions' locations
    if mu_y is None:
        mu_y = np.random.uniform(low=0, high=sh[0], size=n)
    
    if mu_x is None:
        mu_x = np.random.uniform(low=0, high=sh[1], size=n)
    
    # build mask
    mask = build.build_mask(mu_y=mu_y, mu_x=mu_x, sigma=sigma, sh=sh, **kwargs)
    
    # apply mask
    im_out_mat = apply.apply_mask(im=im, mask=mask, bg=bg)
    
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)


def bubbles_conv_mask (im, mu_x=None, mu_y=None, sigma=np.array([5]), bg=0):
    """
    Apply a bubbles mask generated via convolution to a given PIL image.

    Parameters
    ----------
    im : PIL.Image.Image
        The image to apply the bubbles mask to.
    mu_x, mu_y : array_like, optional
        Indices of bubble locations in x (`mu_x`) and y (`mu_y`) dimensions. Must be integers (will be rounded otherwise). The default (`None`) sets to random locations.
    sigma : array_like
        Array of sigmas for the spread of the bubbles. `n` is inferred from the length of this array.
    bg : array_like
        Value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB, or 4 for RGBA.

    Returns
    -------
    im_out : PIL.Image-Image
        Bubble-masked image.
    mask : np.array
        The numpy array used as a mask for `im`, with the same shape as `im`.
    mu_x, mu_y, sigma : np.array
        The locations of bubbles in x (`mu_x`) and y (`mu_y`) dimensions, with corresponding `sigma` values.
    """
    
    n = len(sigma)  # get n bubbles
    sh = np.asarray(im).shape  # get shape
    
    # generate distributions' locations
    if mu_y is None:
        mu_y = np.random.randint(low=0, high=sh[0], size=n)
    
    if mu_x is None:
        mu_x = np.random.randint(low=0, high=sh[1], size=n)
    
    # build mask
    mask = build.build_conv_mask(mu_y=mu_y, mu_x=mu_x, sigma=sigma, sh=sh)
    
    # apply mask
    im_out_mat = apply.apply_mask(im=im, mask=mask, bg=bg)
    
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)

def bubbles_mask_nonzero (im, ref_im=None, sigma=np.array([5]), bg=0, ref_bg=0, max_sigma_from_nonzero=np.inf, **kwargs):
    """
    Apply the bubbles mask to a given PIL image, restricting the possible locations of the bubbles' centres to be within a given multiple of non-zero pixels. The image will be binarised to be im>ref_bg (or ref_im>ref_bg), so binary dilation can be applied, with a slight buffer (rounded to ceiling). The function then picks random locations, and keeps them if the Euclidean distance from the non-zero values is within the tolerance, otherwise rejecting them.
    
    Parameters
    ----------
    im : PIL.Image.Image
        The image to apply the bubbles mask to.
    ref_im : PIL.Image.Image, optional
        The image to be used as the reference image for finding `bg` values. Default is `None` in which case, `im` is used instead.
    sigma : array_like
        Array of sigmas for the spread of the bubbles. `n` is inferred from the length of this array.
    bg : array_like
        Value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB, or 4 for RGBA.
    max_sigma_from_nonzero : array_like
        Maximum multiples of the given sigma value from the nearest nonzero values in ref_im that a bubble's centre can be. Can be `np.inf` for no restriction, or `0` to constrain to within the bounds of `bg` exactly. Can be either one value for all bubbles, or a separate value for each individual bubble. If multiple values are given, then `max_sigma_from_nonzero` must contain the same number of elements as `sigma`.
    **kwargs
        Extra arguments passed to `mask.bubbles_mask` and/or `build.build_mask`, e.g., `scale` and `sum_merge`.

    Returns
    -------
    im_out : PIL.Image-Image
        Bubble-masked image.
    mask : np.array
        The numpy array used as a mask for `im`, with the same shape as `im`.
    mu_x, mu_y, sigma : np.array
        The locations of bubbles in x (`mu_x`) and y (`mu_y`) dimensions, with corresponding `sigma` values.
    """
    sigma = np.array(sigma)
    max_sigma_from_nonzero = np.array(max_sigma_from_nonzero)

    # check that max_sigma_from_nonzero is just one value
    if np.size(max_sigma_from_nonzero) != 1 and np.size(max_sigma_from_nonzero) != np.size(sigma):
        ValueError('max_sigma_from_nonzero should consist of either one element, or the same number of elements as sigma')
    
    # if no limits, just use bubbles_mask()
    if np.all(np.isposinf(max_sigma_from_nonzero)):
        return bubbles_mask(im=im, sigma=sigma, bg=bg, **kwargs)
    
    sh = np.asarray(im).shape  # get shape

    # if no ref_im, use the original image
    if ref_im is None:
        ref_im = im
    else:
        if not np.all(np.asarray(ref_im).shape[:2] == np.asarray(im).shape[:2]):
            ValueError('Inconsistent dimensions between im and ref_im')
    
    # get the acceptable mu locations for each sigma value, and store in `sigma_mu_bounds`
    
    # get acceptable boundaries for each sigma
    sigma_dists = sigma * max_sigma_from_nonzero

    with warnings.catch_warnings():
        # ignore warnings about any invalid casts to integer for np.inf, as this is handled below
        if np.any(np.isposinf(sigma_dists)):
            warnings.filterwarnings("ignore", message='invalid value encountered in cast', category=RuntimeWarning)
        sigma_dil_iters = np.ceil(sigma_dists).astype(int)

    sigma_dil_iters[np.isposinf(sigma_dists)] = 0  # no iterations for bubbles with infinite max dist, as these are added manually below (if all infinite, will have already done an early return via bubbles_mask())
    
    n_iter = np.max(sigma_dil_iters)
    
    ref_im_arr = np.asarray(ref_im)
    max_axis = 1 if ref_im_arr.ndim==2 else 2
    mu_bounds = np.max(ref_im_arr > ref_bg, axis=max_axis)
    
    # this will contain the maximum mu bounds for each sigma, initialised with the acceptable values in case any of sigma_dists==0
    sigma_mu_bounds = [None] * len(sigma)
    
    for i in range(n_iter):
        binary_dilation(mu_bounds, out=mu_bounds)
        
        if i+1 in sigma_dil_iters:
            matching_sigma_idx = list(np.where(np.array(sigma_dil_iters) == (i+1))[0])
            for sigma_i in matching_sigma_idx:
                sigma_mu_bounds[sigma_i] = mu_bounds.copy()

    # if any had infinite distance, all locations will be okay
    for inf_s_idx in np.where(np.isposinf(sigma_dists))[0]:
        sigma_mu_bounds[inf_s_idx] = np.zeros(shape=mu_bounds.shape).astype(bool)
        sigma_mu_bounds[inf_s_idx][:] = True

    # if any had zero distance, the ref_im locations are okay
    for zero_s_idx in np.where(sigma_dists==0)[0]:
        sigma_mu_bounds[zero_s_idx] = np.max(ref_im_arr > ref_bg, axis=max_axis)

    # get possible mu locations for each sigma
    poss_mu_seeds = [np.where(idx_ok) for idx_ok in sigma_mu_bounds]

    mu = np.zeros((2, len(sigma)))
    mu[:] = np.nan

    for i in range(len(sigma)):
        p = np.random.randint(low=0, high=len(poss_mu_seeds[i][0]), size=1)  # index of the seed sample
        mu[:, i] = np.array( [poss_mu_seeds[i][0][p], poss_mu_seeds[i][1][p]] ).flatten()  # store the seed location
        mu[:, i] += np.random.uniform(low=-0.5, high=0.5, size=2)  # add jitter from the seed location, assuming that the pixel indices refer to the centre of the pixels
        dist_i = np.linalg.norm(mu[:, i] - np.array(poss_mu_seeds[i]).T, axis=1)

        if (np.min(dist_i) - 0.5) > sigma_dists[i]:
            ValueError('Exceeded max_dist_from_nonzero!')
    
    # build mask
    mu_y = mu[0, :]
    mu_x = mu[1, :]
    mask = build.build_mask(mu_x=mu_x, mu_y=mu_y, sigma=sigma, sh=sh, **kwargs)
    
    # apply mask
    im_out_mat = apply.apply_mask(im=im, mask=mask, bg=bg)
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)

