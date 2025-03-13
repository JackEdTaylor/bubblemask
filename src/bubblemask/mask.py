from PIL import Image
import numpy as np
from skimage.morphology import binary_dilation
from . import build, apply

def bubbles_mask (im, mu_x=None, mu_y=None, sigma=np.array([5]), bg=0, **kwargs):
    """Apply the bubbles mask to a given PIL image. Returns the edited PIL image, the generated mask, mu_y, mu_x, and sigma.
    
     Keyword arguments:
    im -- the PIL image to apply the bubbles mask to
    mu_x -- x indices (axis 1 in numpy) for bubble locations - set to None (default) for random location
    mu_y -- y indices (axis 0 in numpy) for bubble locations - set to None (default) for random location
    sigma -- array of sigmas for the spread of the bubbles. `n` is inferred from the length of this array
    bg -- value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB, or 4 values, for RGBA
    scale -- should densities' maxima be consistently scaled across different sigma values?
    sum_merge -- should merges, where bubbles overlap, be completed using a simple sum of the bubbles, thresholded to the maxima of the pre-merged bubbles? If False (the default), densities are instead averaged (mean).
    **kwargs -- passed to `build.build_mask`, e.g., `scale` and `sum_merge`.
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
    """Apply a bubbles mask generated via convolution to a given PIL image. Returns the edited PIL image, the generated mask, mu_y, mu_x, and sigma.
    
     Keyword arguments:
    im -- the PIL image to apply the bubbles mask to
    mu_x -- x indices (axis 1 in numpy) for bubble locations - set to None (default) for random location. Must be integers (will be rounded otherwise)
    mu_y -- y indices (axis 0 in numpy) for bubble locations - set to None (default) for random location. Must be integers (will be rounded otherwise)
    sigma -- array of sigmas for the spread of the bubbles. `n` is inferred from the length of this array, but all values should be identical for this method
    bg -- value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB, or 4 values, for RGBA
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

def bubbles_mask_nonzero (im, ref_im=None, sigma=np.array([5]), bg=0, ref_bg=0, max_sigma_from_nonzero=np.inf, max_iter=1e5, **kwargs):
    """Apply the bubbles mask to a given PIL image, restricting the possible locations of the bubbles' centres to be within a given multiple of non-zero pixels. The image will be binarised to be im>ref_bg (or ref_im>ref_bg), so binary dilation can be applied, with a slight buffer (rounded to ceiling). The function then picks random locations, and keeps them if the Euclidean distance from the non-zero values is within the tolerance, otherwise rejecting them. Returns the edited PIL image, the generated mask, mu_y, mu_x, and sigma.
    
     Keyword arguments:
    im -- the image to apply the bubbles mask to
    ref_im -- the image to be used as the reference image for finding the minimum (useful for finding the minimum in a pre-distorted im)
    sigma -- array of sigmas for the spread of the bubbles. `n` is inferred from the length of this array
    bg -- value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB
    max_sigma_from_nonzero -- maximum multiples of the given sigma value from the nearest nonzero values in ref_im that a bubble's centre can be. Can be `np.inf` for no restriction
    max_iter -- maximum number of random samples to try for mu location, for each sigma
    **kwargs -- passed to `mask.bubbles_mask` and/or `build.build_mask`, e.g., `scale` and `sum_merge`.
    """

    # check that max_sigma_from_nonzero is just one value
    if np.size(max_sigma_from_nonzero) != 1:
        ValueError('max_sigma_from_nonzero should be one element')
    
    # if no limits, just use bubbles_mask()
    if np.isposinf(max_sigma_from_nonzero):
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
    sigma_dists = np.array(sigma) * max_sigma_from_nonzero
    sigma_dil_iters = np.ceil(sigma_dists).astype(int)
    
    n_iter = np.max(sigma_dil_iters)
    
    ref_im_arr = np.asarray(ref_im)
    max_axis = 1 if ref_im_arr.ndim==2 else 2
    mu_bounds = np.max(ref_im_arr > ref_bg, axis=max_axis)
    
    # this will contain the maximum mu bounds for each sigma
    sigma_mu_bounds = [None] * len(sigma)
    
    for i in range(n_iter):
        binary_dilation(mu_bounds, out=mu_bounds)
        
        if i+1 in sigma_dil_iters:
            matching_sigma_idx = list(np.where(np.array(sigma_dil_iters) == (i+1))[0])
            for sigma_i in matching_sigma_idx:
                sigma_mu_bounds[sigma_i] = mu_bounds.copy()

    # get possible mu locations for each sigma
    poss_mu_seeds = [np.where(idx_ok) for idx_ok in sigma_mu_bounds]

    mu = np.zeros((2, len(sigma)))
    mu[:] = np.nan

    for i in range(len(sigma)):
        attempt = 0
        sample_ok = False

        while not sample_ok:
            attempt += 1

            if attempt > max_iter:
                RuntimeError('Reached max_iter!')

            p = np.random.randint(low=0, high=len(poss_mu_seeds[i][0]), size=1)  # index of the seed sample
            mu[:, i] = np.array( [poss_mu_seeds[i][0][p], poss_mu_seeds[i][1][p]] ).flatten()  # store the seed location
            mu[:, i] += np.random.uniform(low=-0.5, high=0.5, size=2)  # add jitter from the seed location
            dist_i = np.linalg.norm(mu[:, i] - np.array(poss_mu_seeds[i]).T, axis=1)

            # reject sample if outside of bounds
            if np.min(dist_i) > sigma_dists[i]:
                mu[:, i] = np.nan
            else:
                sample_ok = True
    
    # build mask
    mu_y = mu[0, :]
    mu_x = mu[1, :]
    mask = build.build_mask(mu_x=mu_x, mu_y=mu_y, sigma=sigma, sh=sh, **kwargs)
    
    # apply mask
    im_out_mat = apply.apply_mask(im=im, mask=mask, bg=bg)
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)

