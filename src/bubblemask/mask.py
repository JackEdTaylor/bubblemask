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


def bubbles_mask_nonzero (im, ref_im=None, sigma = np.array([5]), bg=0, ref_bg=0, max_sigma_from_nonzero=np.inf, **kwargs):
    """Apply the bubbles mask to a given PIL image, restricting the possible locations of the bubbles' centres to be within a given multiple of non-zero pixels. The image will be binarised to be im>ref_bg (or ref_im>ref_bg), so binary dilation can be applied. Any products of sigma and max_sigma_from_nonzero that are floats will be rounded to the nearest integer. Returns the edited PIL image, the generated mask, mu_y, mu_x, and sigma.
    
     Keyword arguments:
    im -- the image to apply the bubbles mask to
    ref_im -- the image to be used as the reference image for finding the minimum (useful for finding the minimum in a pre-distorted im)
    sigma -- array of sigmas for the spread of the bubbles. `n` is inferred from the length of this array
    bg -- value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB
    scale -- should densities' maxima be consistently scaled across different sigma values?
    sum_merge -- should merges, where bubbles overlap, be completed using a simple sum of the bubbles, thresholded to the maxima of the pre-merged bubbles? If False (the default), densities are instead averaged (mean).
    max_sigma_from_nonzero -- maximum multiples of the given sigma value from the nearest nonzero values in ref_im that a bubble's centre can be. Can be `np.inf` for no restriction
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
    
    # get the acceptable mu locations for each sigma value, and store in `sigma_mu_bounds`
    
    # get acceptable boundaries for each sigma
    sigma_dil_iters = np.round(np.array(sigma) * max_sigma_from_nonzero).astype(int)

    # check that the product of sigma and max_sigma_from_nonzero is always an integer - otherwise give a warning
    if np.any(sigma_dil_iters != sigma_dil_iters.round()):
        Warning('Some values in max_sigma_from_nonzero*sigma are non-integer. These will be rounded to the nearest integer!')
    
    n_iter = np.max(sigma_dil_iters)
    
    ref_im_arr = np.asarray(ref_im)
    max_axis = 1 if ref_im_arr.ndim==2 else 2
    mu_bounds = np.max(ref_im_arr > ref_bg, axis=max_axis)
    
    # this will contain the desired mu bounds for each sigma
    sigma_mu_bounds = [None] * len(sigma)
    
    for i in range(n_iter):
        binary_dilation(mu_bounds, out=mu_bounds)
        
        if i+1 in sigma_dil_iters:
            matching_sigma_idx = list(np.where(np.array(sigma_dil_iters) == (i+1))[0])
            for sigma_i in matching_sigma_idx:
                sigma_mu_bounds[sigma_i] = mu_bounds.copy()

    # get possible mu locations for each sigma
    poss_mu = [np.where(idx_ok) for idx_ok in sigma_mu_bounds]
    
    # get mu locations for each bubble, as an index in the possible mu values
    mu_idx = [np.random.randint(low=0, high=len(x[0]), size=1) for x in poss_mu]
    
    # generate actual mu values as index plus uniform noise between -0.5 and 0.5 (rather than all mus being on integers)
    mu_y = [int(poss_mu[i][0][mu_idx[i]]) for i in range(len(poss_mu))] + np.random.uniform(low=-0.5, high=0.5, size=len(mu_idx))
    mu_x = [int(poss_mu[i][1][mu_idx[i]]) for i in range(len(poss_mu))] + np.random.uniform(low=-0.5, high=0.5, size=len(mu_idx))
    
    # build mask
    mask = build.build_mask(mu_y=mu_y, mu_x=mu_x, sigma=sigma, sh=sh, **kwargs)
    
    # apply mask
    im_out_mat = apply.apply_mask(im=im, mask=mask, bg=bg)
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)
