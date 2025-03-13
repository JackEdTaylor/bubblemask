import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

def build_mask(mu_y, mu_x, sigma, sh, scale=True, sum_merge=False):
    """
    Build a Bubbles mask which can be applied to an image of shape `sh`.
    
    Parameters
    ----------
    mu_y : array_like
        The locations of the bubbles' centres, in numpy axis 0.
    mu_x : array_like
        The locations of the bubbles centres, in numpy axis 1 (should be same len as `mu_y`).
    sigma : array_like
        Array of sigmas for the spread of the bubbles (should be same len as `mu_y`).
    sh : tuple
        Shape (np.shape) of the desired mask (usually the shape of the respective image).
    scale : bool
        Should densities' maxima be consistently scaled across different sigma values? Default is `True`.
    sum_merge : bool
        Should merges, where bubbles overlap, be completed using a simple sum of the bubbles, thresholded to the maxima of the pre-merged bubbles? If False (the default), densities are instead averaged (mean).

    Returns
    -------
    mask : np.array
        The mask, with shape `sh`.
    """
    # check lengths match and are all 1d
    gauss_pars_sh = [np.shape(x) for x in [mu_y, mu_x, sigma]]
    gauss_pars_n_dims = [len(x) for x in gauss_pars_sh]
    
    if len(set(gauss_pars_sh))!=1 or any(gauss_pars_n_dims)!=1:
        ValueError('mu_y, mu_x, and sigma should all be 1-dimensional arrays of identical length')
    
    # for each distribution, generate the bubble
    dists = [
        # get the outer product of vectors for the densities of pixel indices across x and y dimensions, for each distribution (provides 2d density)
        np.outer(
            norm.pdf(np.arange(stop=sh[0]), mu_y_i, sigma_i),
            norm.pdf(np.arange(stop=sh[1]), mu_x_i, sigma_i)
            )
        for mu_x_i, mu_y_i, sigma_i in zip(mu_x, mu_y, sigma)
        ]
    
    # scale all bubbles consistently if requested
    if scale:
        dists = [x/np.max(x) for x in dists]
    
    if sum_merge:
        # sum the distributions, then threshold the maximum to the maximum peak
        mask = np.sum(dists, axis=0)
        mask[mask>np.max(dists)] = np.max(dists)

    else:
        # merge using average of densities
        mask = np.mean(dists, axis=0)
    
    # scale density to within [0, 1] (will already be scaled to [0, 1] above if scale==True)
    mask /= np.max(mask)
    
    return(mask)

def build_conv_mask(mu_y, mu_x, sigma, sh):
    """
    Build a Bubbles mask via convolution which can be applied to an image of shape `sh`. Unlike build_mask(), build_conv_mask() requires that all sigma values are equal.
    
    Parameters
    ----------
    mu_y : array_like
        The locations of the bubbles' centres, in numpy axis 0. Must be integers (will be rounded otherwise).
    mu_x : array_like
        The locations of the bubbles centres, in numpy axis 1 (should be same len as `mu_y`).  Must be integers (will be rounded otherwise).
    sigma : array_like
        A single value for sigma, or else an array of sigmas for the spread of the bubbles (in which case, should be same len as mu_y, and should all be identical).
    sh : tuple
        Shape (np.shape) of the desired mask (usually the shape of the respective image).

    Returns
    -------
    mask : np.array
        The mask, with shape `sh`.
    """
    # if sigma is given as a list, get the single value
    if isinstance(sigma, list) | isinstance(sigma, np.ndarray):
        sigma = np.unique(sigma)
    
    # if more than one sigma value, give error
    if len(sigma)>1:
        ValueError('for the convolution approach, sigma should be of length one, or else all values should be identical')

    # check lengths for mu match and are both 1d
    gauss_pars_sh = [np.shape(x) for x in [mu_y, mu_x]]
    gauss_pars_n_dims = [len(x) for x in gauss_pars_sh]
    
    if len(set(gauss_pars_sh))!=1 or any(gauss_pars_n_dims)!=1:
        ValueError('mu_y and mu_x should both be 1-dimensional arrays of identical length')

    # generate the pre-convolution mask
    mask_preconv = np.zeros(sh)

    mask_preconv[
        np.array(mu_y).astype(int),
        np.array(mu_x).astype(int)
        ] = 1

    # apply the filter via scipy.signal.gaussian_filter (uses a series of 1d convolutions)
    mask = gaussian_filter(mask_preconv, sigma=float(sigma), mode='constant', cval=0.0)

    # scale the mask
    mask /= np.max(mask)

    return(mask)
