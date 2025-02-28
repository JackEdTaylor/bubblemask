from argparse import ArgumentParser
from . import mask
import numpy as np
from PIL import Image

if __name__ == "__main__":    
    parser = ArgumentParser(prog='bubblemask')
    
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
    im_out = mask.bubbles_mask(im=im, mu_x=args.mu_x, mu_y=args.mu_y, sigma=args.sigma, bg=args.background, scale=args.unscaled, sum_merge=args.summerge)[0]
    im_out.save(args.output)
