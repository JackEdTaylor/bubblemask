from PIL import Image
import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_dilation

def build_mask(mu_y, mu_x, sigma, sh, scale, sum_merge):
    """Build a Bubbles mask which can be applied to an image of shape `sh`. Returns a matrix for the mask.
    
     Keyword arguments:
    mu_y -- the locations of the bubbles centres, in numpy axis 0
    mu_x -- the locations of the bubbles centres, in numpy axis 1 (should be same len as mu_y)
    sigma -- array of sigmas for the spread of the bubbles (should be same len as mu_y)
    sh -- shape (np.shape) of the desired mask (usually the shape of the respective image)
    scale -- should densities' maxima be consistently scaled across different sigma values?
    sum_merge -- should merges, where bubbles overlap, be completed using a simple sum of the bubbles, thresholded to the maxima of the pre-merged bubbles? If False (the default), densities are instead averaged (mean).
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
    Build a Bubbles mask via convolution which can be applied to an image of shape `sh`. Returns a matrix for the mask.
    Unlike build_mask(), build_conv_mask() requires that all sigma values are equal.
    
     Keyword arguments:
    mu_y -- the locations of the bubbles centres, in numpy axis 0. Must be integers (will be rounded otherwise)
    mu_x -- the locations of the bubbles centres, in numpy axis 1 (should be same len as mu_y). Must be integers (will be rounded otherwise)
    sigma -- a single value for sigma, or else an array of sigmas for the spread of the bubbles (in which case, should be same len as mu_y, and should all be identical)
    sh -- shape (np.shape) of the desired mask (usually the shape of the respective image)
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


def apply_mask(im, mask, bg=0):
    """Apply a mask to image `im`. Returns a PIL image.
    
     Keyword arguments:
    im -- the original image
    mask -- the mask to apply to the image
    bg -- value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB, or 4 values for RGBA
    """
    
    sh = np.asarray(im).shape
    
    if len(sh)>2:
        n_col_chs = sh[2]
    else:
        n_col_chs = 1
    
    if n_col_chs > 1:
        im_out_mat = im * np.repeat(mask[:,:,np.newaxis], n_col_chs, axis=2)
    else:
        im_out_mat = im * mask
    
    # adjust the background
    if np.any(bg != 0):
        if n_col_chs > 1:
            im_bg_mat = bg * (1 - np.repeat(mask[:,:,np.newaxis], sh[2], axis=2))
        else:
            im_bg_mat = bg * (1 - mask)
        
        im_out_mat += im_bg_mat
    
    return(im_out_mat)


def bubbles_mask (im, mu_x=None, mu_y=None, sigma=np.repeat(25, repeats=5), bg=0, scale=True, sum_merge=False):
    """Apply the bubbles mask to a given PIL image. Returns the edited PIL image, the generated mask, mu_y, mu_x, and sigma.
    
     Keyword arguments:
    im -- the PIL image to apply the bubbles mask to
    mu_x -- x indices (axis 1 in numpy) for bubble locations - set to None (default) for random location
    mu_y -- y indices (axis 0 in numpy) for bubble locations - set to None (default) for random location
    sigma -- array of sigmas for the spread of the bubbles. `n` is inferred from this array
    bg -- value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB, or 4 values, for RGBA
    scale -- should densities' maxima be consistently scaled across different sigma values?
    sum_merge -- should merges, where bubbles overlap, be completed using a simple sum of the bubbles, thresholded to the maxima of the pre-merged bubbles? If False (the default), densities are instead averaged (mean).
    """
    
    n = len(sigma)  # get n bubbles
    sh = np.asarray(im).shape  # get shape
    
    # generate distributions' locations
    if mu_y is None:
        mu_y = np.random.uniform(low=0, high=sh[0], size=n)
    
    if mu_x is None:
        mu_x = np.random.uniform(low=0, high=sh[1], size=n)
    
    # build mask
    mask = build_mask(mu_y=mu_y, mu_x=mu_x, sigma=sigma, sh=sh, scale=scale, sum_merge=sum_merge)
    
    # apply mask
    im_out_mat = apply_mask(im=im, mask=mask, bg=bg)
    
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)


def bubbles_conv_mask (im, mu_x=None, mu_y=None, sigma=np.repeat(25, repeats=5), bg=0):
    """Apply a bubbles mask generated via convolution to a given PIL image. Returns the edited PIL image, the generated mask, mu_y, mu_x, and sigma.
    
     Keyword arguments:
    im -- the PIL image to apply the bubbles mask to
    mu_x -- x indices (axis 1 in numpy) for bubble locations - set to None (default) for random location. Must be integers (will be rounded otherwise)
    mu_y -- y indices (axis 0 in numpy) for bubble locations - set to None (default) for random location. Must be integers (will be rounded otherwise)
    sigma -- array of sigmas for the spread of the bubbles. `n` is inferred from this array, but all values should be identical for this method
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
    mask = build_conv_mask(mu_y=mu_y, mu_x=mu_x, sigma=sigma, sh=sh)
    
    # apply mask
    im_out_mat = apply_mask(im=im, mask=mask, bg=bg)
    
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)


def bubbles_mask_nonzero (im, ref_im=None, sigma = np.repeat(25, repeats=5), bg=0, scale=True, sum_merge=False, max_sigma_from_nonzero=np.inf):
    """Apply the bubbles mask to a given PIL image, restricting the possible locations of the bubbles' centres to be within a given multiple of non-zero pixels. The image will be binarised to be im<=bg gives 0, else 1, so binary dilation can be applied. Returns the edited PIL image, the generated mask, mu_y, mu_x, and sigma.
    
     Keyword arguments:
    im -- the image to apply the bubbles mask to
    ref_im -- the image to be used as the reference image for finding the minimum (useful for finding the minimum in a pre-distorted im)
    sigma -- array of sigmas for the spread of the bubbles. `n` is inferred from this array
    bg -- value for the background, from 0 to 255. Can also be an array of 3 values from 0 to 255, for RGB
    scale -- should densities' maxima be consistently scaled across different sigma values?
    sum_merge -- should merges, where bubbles overlap, be completed using a simple sum of the bubbles, thresholded to the maxima of the pre-merged bubbles? If False (the default), densities are instead averaged (mean).
    max_sigma_from_nonzero -- maximum multiples of the given sigma value from the nearest nonzero (in practice, non-minimum) values that a bubble's centre can be. Can be `np.inf` for no restriction
    """

    # check that max_sigma_from_nonzero is just one value
    if np.size(max_sigma_from_nonzero) != 1:
        ValueError('max_sigma_from_nonzero should be one element')
    
    sh = np.asarray(im).shape  # get shape
    
    # if no limits, just use bubbles_mask()
    if np.isposinf(max_sigma_from_nonzero):
        return(bubbles_mask(im=im, sigma=sigma, bg=bg, scale=scale))
    
    # get the acceptable mu locations for each sigma value, and store in `sigma_mu_bounds`
    
    # get acceptable boundaries for each sigma
    sigma_dil_iters = [int(np.round(s * max_sigma_from_nonzero)) for s in sigma]
    
    n_iter = max(sigma_dil_iters)
    
    if ref_im is None:
        mu_bounds = np.max(np.asarray(im) > bg, axis=2)
    else:
        mu_bounds = np.max(np.asarray(ref_im) > bg, axis=2)
    
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
    mask = build_mask(mu_y=mu_y, mu_x=mu_x, sigma=sigma, sh=sh, scale=scale, sum_merge=sum_merge)
    
    # apply mask
    im_out_mat = apply_mask(im=im, mask=mask, bg=bg)
    im_out = Image.fromarray(np.uint8(im_out_mat))
    
    return(im_out, mask, mu_x, mu_y, sigma)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument('-i', '--input', help='the file path for the input image',
                        action='store', required=True, type=str)
    
    parser.add_argument('-o', '--output', help='the path of the desired output file',
                        action='store', required=True, type=str)
    
    parser.add_argument('-s', '--sigma', nargs='+', help='a list of sigmas for the bubbles, in space-separated format (e.g., "10 10 15")',
                        action='store', required=True, type=float)
    
    parser.add_argument('-x', '--mu_x', nargs='+', help='x indices (axis 1 in numpy) for bubble locations, in space-separated format - leave blank (default) for random location', type=float)
    
    parser.add_argument('-y', '--mu_y', nargs='+', help='y indices (axis 0 in numpy) for bubble locations, in space-separated format - leave blank (default) for random location', type=float)
    
    parser.add_argument('-b', '--background', nargs='+', help='the desired background for the image, as a single integer from 0 to 255 (default=0), or space-separated values for each channel in the image',
                        action='store', type=int, default=0)
    
    parser.add_argument('--unscaled', help='do not scale the densities of the bubbles to have the same maxima',
                        action='store_false')
    
    parser.add_argument('--summerge', help='sum_merge -- should merges, where bubbles overlap, be completed using a simple sum of the bubbles, thresholded to the maxima of the pre-merged bubbles? If not (the default), densities are instead averaged (mean).',
                        action='store_true')
    
    parser.add_argument('--seed', help='random seed to use', action='store', type=int)
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    im = Image.open(args.input)
    im_out = bubbles_mask(im=im, mu_x=args.mu_x, mu_y=args.mu_y, sigma=args.sigma, bg=args.background, scale=args.unscaled, sum_merge=args.summerge)[0]
    im_out.save(args.output)
    
    
