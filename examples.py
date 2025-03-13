# %% Setup

from PIL import Image
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.bubblemask import mask, build
import time
from tqdm import trange

np.random.seed(152872)

in_dir = op.join('img', 'pre')
out_dir = op.join('img', 'post')

# %% Example 1 - face

face = Image.open(op.join(in_dir, 'face.png'))

face1, mask1, mu_x, mu_y, sigma = mask.bubbles_mask(im=face, sigma=[17, 19, 20.84, 25, 30], bg=127)

print(mu_x)
print(mu_y)
print(sigma)

face1.save(op.join(out_dir, 'face1.png'))

fig = plt.figure(figsize=(4.35, 5))
plt.imshow(mask1)
plt.colorbar()
fig.savefig(op.join(out_dir, 'face1_mask.png'), dpi=100, bbox_inches='tight')

# %% Example 2 - face's eyes

face2 = mask.bubbles_mask(im=face, mu_x=[85, 186.7], mu_y=[182.5, 182.5], sigma=[20, 10], bg=127)[0]
face2.save(op.join(out_dir, 'face2.png'))

# %% Example 3 - compare to convolution method

mu_x = [70, 21, 47, 254, 193]
mu_y = [190, 102, 219, 63, 80]
sigma = [20,20,20,20,20]

face3a, mask3a, _, _, _ = mask.bubbles_mask(im=face, mu_x=mu_x, mu_y=mu_y, sigma=sigma, bg=127)
face3b, mask3b, _, _, _ = mask.bubbles_conv_mask(im=face, mu_x=mu_x, mu_y=mu_y, sigma=sigma, bg=127)

face3a.save(op.join(out_dir, 'face3a.png'))
face3b.save(op.join(out_dir, 'face3b.png'))

fig = plt.figure(figsize=(4.35, 5))
plt.imshow(mask3a)
plt.colorbar()
fig.savefig(op.join(out_dir, 'face3a_mask.png'), dpi=80, bbox_inches='tight')

fig = plt.figure(figsize=(4.35, 5))
plt.imshow(mask3b)
plt.colorbar()
fig.savefig(op.join(out_dir, 'face3b_mask.png'), dpi=80, bbox_inches='tight')

fig = plt.figure(figsize=(4.35, 5))
plt.imshow(mask3a-mask3b)
plt.colorbar()
fig.savefig(op.join(out_dir, 'face3_mask_diff.png'), dpi=80, bbox_inches='tight')

# %% Example 3 - letter a on the cat image

a_cat = Image.open(op.join(in_dir, 'a_cat.png'))
a_cat_ref = Image.open(op.join('img', 'pre', 'a_cat_ref.png'))

a_cat1, maskacat1, mu_x, mu_y, sigma = mask.bubbles_mask_nonzero(im=a_cat, ref_im=a_cat_ref, sigma=[10,10,10,10,10], max_sigma_from_nonzero=1, ref_bg=[0,0,0,255], bg=[0,0,0,255])

a_cat1.save(op.join(out_dir, 'a_cat1.png'))

# demonstrate that the space is unused
a_cat2, maskacat2, mu_x, mu_y, sigma = mask.bubbles_mask_nonzero(im=a_cat, ref_im=a_cat_ref, sigma=np.repeat(2.5, repeats=1000), max_sigma_from_nonzero=5, ref_bg=[0,0,0,255], bg=[0,0,0,255])

a_cat2.save(op.join(out_dir, 'a_cat2.png'))

fig = plt.figure(figsize=(6, 2.75))
plt.imshow(maskacat2)
plt.colorbar()
fig.savefig(op.join(out_dir, 'a_cat2_mask.png'), dpi=80, bbox_inches='tight')

# demonstrate that each bubble can have a different max_sigma_from_nonzero value, and that 0 and np.inf are supported
a_cat3, maskacat3, mu_x, mu_y, sigma = mask.bubbles_mask_nonzero(im=a_cat, ref_im=a_cat_ref, sigma=[25, 10, 5], max_sigma_from_nonzero=[np.inf, 2.75, 0], ref_bg=[0,0,0,255], bg=[0,0,0,255])

a_cat3.save(op.join(out_dir, 'a_cat3.png'))

# %% Example 4 - cat

cat = Image.open(op.join(in_dir, 'cat.png'))
cat1 = mask.bubbles_mask(im=cat, sigma=np.repeat(10, 20), bg=127)[0]
cat2 = mask.bubbles_mask(im=cat, sigma=np.repeat(10, 20), bg=[127, 0, 127])[0]
cat3 = mask.bubbles_mask(im=cat.convert('RGBA'), sigma=np.repeat(10, 20), bg=[0, 0, 0, 0])[0]

cat1.save(op.join(out_dir, 'cat1.png'))
cat2.save(op.join(out_dir, 'cat2.png'))
cat3.save(op.join(out_dir, 'cat3.png'))

# %% Example 5 - masks

# same bubble parameters for all masks
mu_y = [20, 30, 70]
mu_x = [20, 30, 90]
sigma = [5, 10, 7.5]
sh = (100, 100)

masks = [build.build_mask(mu_y, mu_x, sigma, sh, scale=True, sum_merge=False),
         build.build_mask(mu_y, mu_x, sigma, sh, scale=True, sum_merge=True),
         build.build_mask(mu_y, mu_x, sigma, sh, scale=False, sum_merge=False),
         build.build_mask(mu_y, mu_x, sigma, sh, scale=False, sum_merge=True)]

for i in range(4):
    fig = plt.figure(figsize=(3, 2.5))
    plt.imshow(masks[i], interpolation=None)
    plt.colorbar()
    fig.savefig(op.join(out_dir, f'mask{i+1}.png'), dpi=100, bbox_inches='tight')

# %%
# Compare timing for convolution and outer product approaches

n_iter = 50  # per combination of parameters

sizes = np.array([25, 50, 100, 250, 500, 1000, 1500, 2000])

n_bubbles = np.array([1, 5, 10, 100])

sigmas = np.array([1, 5, 10, 25])

conv_times = np.zeros((len(sizes), len(n_bubbles), len(sigmas), n_iter))
op_times = conv_times.copy()

for i in trange(n_iter, desc='Timing approaches'):
    for sz in range(len(sizes)):
        # get shape
        sh = (sizes[sz], sizes[sz])

        for nb in range(len(n_bubbles)):

            for sg in range(len(sigmas)):
                # select locations from uniform distribution of integers
                # (convolution approach requires integers)
                mu_x = np.random.randint(sizes[sz], size=n_bubbles[nb])
                mu_y = np.random.randint(sizes[sz], size=n_bubbles[nb])

                # time convolution approach
                conv_s = time.process_time()
                conv_m = build.build_conv_mask(mu_x=mu_x, mu_y=mu_y, sigma=[sigmas[sg]], sh=sh)
                conv_e = time.process_time()
                conv_times[sz, nb, sg, i] = conv_e - conv_s
                # time outer product approach
                op_s = time.process_time()
                op_m = build.build_mask(mu_x=mu_x, mu_y=mu_y, sigma=[sigmas[sg]], sh=sh, scale=True, sum_merge=False)
                op_e = time.process_time()
                op_times[sz, nb, sg, i] = op_e - op_s

# %%
# plot timing results
fig, axs = plt.subplots(1, len(sigmas), figsize=(6.5, 2.5))

cmap = mpl.colormaps.get_cmap('Dark2')

for sg, sigma in enumerate(sigmas):
    axs[sg].set_title(f'$\sigma$={sigma}')
    # axs[sg].axhline(y=0, linestyle='--', color='k')

    for nb, n_bub in enumerate(n_bubbles):
        axs[sg].plot(sizes, np.mean( conv_times[:, nb, sg, :]*1000, axis=1 ), color=cmap(nb), linestyle='dashed')
        axs[sg].plot(sizes, np.mean( op_times[:, nb, sg, :]*1000, axis=1 ), color=cmap(nb))
    
    axs[sg].set_ylim(0, max([np.max(conv_times), np.max(op_times)])*1000)

# create legends
lines_col = [mpl.lines.Line2D([0], [0], color=cmap(nb)) for nb in range(len(n_bubbles))]
lines_style = [mpl.lines.Line2D([0], [0], linestyle=st, color='k') for st in ['solid', 'dashed']]

fig.legend(lines_col, n_bubbles, title='N Bubbles', bbox_to_anchor=(0.075, 1.02, 0.45, 0.2), loc='lower left', mode='expand', borderaxespad=0, ncol=len(n_bubbles))
fig.legend(lines_style, ['Density', 'Convolution'], title='Bubble Method', bbox_to_anchor=(0.575, 1.02, 0.375, 0.2), loc='lower left', mode='expand', borderaxespad=0, ncol=len(n_bubbles))

fig.supxlabel('Image Size (px$^2$)')
fig.supylabel('Time Taken (ms)')

fig.tight_layout()

fig.savefig(op.join(out_dir, f'timing_comparison.png'), dpi=300, bbox_inches='tight')
