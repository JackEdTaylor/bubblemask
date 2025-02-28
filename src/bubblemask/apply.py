import numpy as np

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
