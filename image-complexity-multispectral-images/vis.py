"""
Implements several functions useful for visualization of images, particularly color distributions.
"""

import os, sys, numpy as np, skimage, argparse, scipy, matplotlib
from skimage import io, color as skcolor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import imdisplay, load_helper
from mpl_toolkits.mplot3d import Axes3D, axes3d


def main_single_display(args):
    assert os.path.exists(args.input), "Input " + args.input + " does not exist."
    img, R, G, B, orig_mask = load_helper(args.input, blur_sigma=args.blur)

    #---------------------------------------------------------------------------------#

    ### Show the original image ###
    if args.show_img: display_orig(img, orig_mask, args)
    ### Show HSV decomposition ###
    if args.show_hsv: display_hsv(img, orig_mask, args)
    ### Polar Hue histogram from HSV ###
    if args.hist_hsv_polar: plot_polar_hsv(img, orig_mask, args)
    ### Show Scalar colormap index image ###
    if args.show_twilight_img: # TODO other maps
        plot_colour_mapped_scalar_image(img, orig_mask, args, 'twilight')
    ### Polar Colour histogram from named colormaps ###
    if args.hist_twilight_polar: # TODO other maps
        plot_polar_generic(img, orig_mask, args, 'twilight')
    ### 1D RGB histograms ###
    if args.hist_rgb_1d: plot_1D_rgb(R, G, B)
    ### 3D RGB histograms ###
    if args.hist_rgb_3d: plot_3D_rgb(R, G, B)
    ### Plot colored pixels with density projections ###
    if args.scatter_densities: plot_density_proj(R, G, B)
    ### Display the manually defined unfolded 1D color histogram ###
    if args.manual_unfolded_1d: plot_manual_unfolded_1d(R, G, B)
    ### Show projection of pixel values ###
    if args.projected_pixels: plot_projected_pixels(R, G, B)

    #> Final display activation <#
    plt.show()

def main_dir_write(args):
    if args.write_1d_histo_vals:
        folder = args.input
        output_filename = args.output_file
        write_manual_unfolded_1d(folder, output_filename)

def plot_projected_pixels(R, G, B, subsampling=8000, verbose=True):
    # Put image pixels into normalized 3D RGB colour space
    print('Generating plot of projected pixels')
    algo = 'pca' # 'tsne', 'pca', 'lle'
    n_pixels = len(R)
    P = np.vstack( (R, G, B) ).T / 255.0
    from utils import plotDimensionallyReducedVectorsIn2D
    if n_pixels > subsampling:
        if verbose: print('Subsampling for projection calculation')
        rindsd = np.random.choice(n_pixels, size=subsampling, replace=False)
        P = P[rindsd, :]
    plotDimensionallyReducedVectorsIn2D(P, method=algo, colors=P)

def write_manual_unfolded_1d(folder, output_filename, verbose=True):
    """
    Writes a CSV file with the histogram values for each image file in the specified folder.

    The first two rows are special:
        The first row shows the titles of columns.
        The second row holds the tuple colors per bin.
    Each row after that holds the bin counts for each image, with the first column holding the filename.
    """
    #assert os.path.isdir(folder), "Input " + folder + " must be a directory"
    
    assert not os.path.exists(output_filename), "Output filename " + output_filename + " already exists"
    def append_to_file(string):
        if not string.endswith("\n"): string += '\n'
        with open(output_filename, "a") as file_handle:
            file_handle.write(string)
            file_handle.flush()
    exts  = ['.png', '.jpg']
    exts += [ _e.upper() for _e in exts ]
    if os.path.isdir(folder):
        images = [os.path.join(folder,f) for f in os.listdir(folder) if any(f.endswith(ext) for ext in exts)]
    else:
        images = [folder] # folder is actually just an img file
    N = len(images)
    print('Generating CSV file (for %d files). Writing to' % N, output_filename)
    prev_bins, prev_colours = None, None
    for i, image in enumerate(images):
        print('On %d / %d (%.1f%%)' % (i, N, i*100/N))
        img, R, G, B, orig_mask = load_helper(image)
        # Get: (1D pseudocolors per pixel, histogram values per bin, bin edge values, bin edge color values)
        P_1d, H, bins, bin_edge_colours = plot_manual_unfolded_1d(R, G, B, make_plot=False)
        if verbose:
            print("\tShapes", P_1d.shape, H.shape, bins.shape, bin_edge_colours.shape)
        if prev_bins is None and prev_colours is None:
            # First rows
            titles = 'name,' + ",".join(['Bin_%d' % j for j in range(1,len(bins))])
            append_to_file(titles)
            bin_middle_colors = 0.5 * (bin_edge_colours[1:, :] + bin_edge_colours[:-1, :])
            colors_row = 'bin_colors,' + ",".join([str(clr) for clr in bin_middle_colors])
            append_to_file(colors_row)
            prev_bins, prev_colours = bins, bin_edge_colours
        else:
            # Check for consistency among first few images
            # Bin edge locations (in 1D) and the corresponding bin-middle colours (in 3D) should be unchanged
            if i < 5:
                assert ((bins - prev_bins)**2).sum() < 1e-6
                assert ((bin_edge_colours - prev_colours)**2).sum() < 1e-6
                prev_bins, prev_colours = bins, bin_edge_colours
        # Write values for this image to the CSV
        if verbose: print('\tH:', H)
        img_histo_vals = image + "," + ",".join([str(hs) for hs in H])
        append_to_file(img_histo_vals)

def plot_manual_unfolded_1d(R, G, B, cmap_name_or_index=2, nbins=75, verbose=False, n_search_bins=200, kde_bins=150,
                            add_kde_curve=True, logscale=False, make_plot=True):
    """
    Plots a 1D histogram of colour for the given image, unfolded in a manually defined fashion.
    Any matplotlib colormap is supported:
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    """
    assert np.max(R) > 1.0 or np.max(G) > 1.0 or np.max(B) > 1.0, "Unexpected non-integer input received"
    print('Generating plot of manually unfolded 1D colour histogram')
    # Choose colormap C: [0,1] --> R^4 = (r,g,b,a)
    # TODO It is possible to define a custom colormap simply via a function that linearly interpolates between
    # chosen colors at specified positions, but that is not implemented here.
    indexed_choices = ['nipy_spectral', 'gist_ncar', 'custom']
    if type(cmap_name_or_index) is int:
        cmap_name = indexed_choices[cmap_name_or_index]
    else:
        cmap_name = cmap_name_or_index
    if cmap_name == 'custom':
        cmap_name = 'Manually Defined'
        from utils import color_multiinterpolator, color_biinterpolator, from_ints
        colorlist = [
            from_ints(0,0,0),           #
            from_ints(137, 51, 163),    #
            from_ints(38, 39, 228),     #
            from_ints(83, 206, 249),    #
            from_ints(83, 249, 197),    #
            from_ints(13, 149, 9),      #
            from_ints(173, 219, 59),    #
            from_ints(240, 249, 18),    #
            from_ints(255, 174, 5),     #
            from_ints(140, 98, 37),     #
            from_ints(249, 7, 9),       #
            from_ints(251, 138, 234),   #
            from_ints(230, 230, 230),   #
            from_ints(255, 255, 255)    #
        ]
        # For times, 0 and 1 will be prepended and appended
        times = [
            0.12,  # violet
            0.20,  # blue
            0.24,  # light blue
            0.27,  # turqoise/coral
            0.40,  # green
            0.45,  # green-yellow mix
            0.50,  # yellow
            0.60,  # orange
            0.70,  # brown
            0.80,  # red
            0.90,  # pink
            0.96   # light grey
        ]
        #times = np.linspace(0,1,len(colorlist))[1:-1]
        #print(times)
        C = color_multiinterpolator(colorlist, times)
    else:
        C = plt.cm.get_cmap(cmap_name)
    # Use the colormap to get the bin values
    bin_edge_indices = np.linspace(0.0, 1.0, n_search_bins)
    bin_middle_inds = (bin_edge_indices[1:] + bin_edge_indices[:-1]) / 2.0
    bin_middle_colors = np.array([ C(midind)[0:3] for midind in bin_middle_inds ]) # remove alpha after mapping
    if verbose: print(bin_middle_inds, '\n', bin_middle_colors, '\n', bin_middle_inds.shape, bin_middle_colors.shape)
    # Put image pixels into normalized 3D RGB colour space
    P = np.vstack( (R, G, B) ).T / 255.0
    # We want to map each 3D colour to its 1D representation to get a histogram of it
    # Let's get the nearest neighbour of each p in P, from within the set of mapped
    # middle bin colors with a KD-tree
    from scipy import spatial
    if verbose: print('Building and querying KD-tree')
    tree = spatial.cKDTree(bin_middle_colors)
    distances, neb_indices = tree.query(P, k=1)
    if verbose: print('Distance max and avg', np.mean(distances), np.max(distances))
    # Assign 1D pseudo-colour values to each position in the colour point cloud of the image
    single_dim_P = bin_middle_inds[ neb_indices ]
    if make_plot:
        # Generate 1D histogram
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(single_dim_P, bins=nbins, density=True, range=(0,1),
                                        linewidth=1.1, edgecolor='black')
        bin_centers = 0.5 * (bins[:-1] + bins[1:]) # Histo bins, not search bins
        col = bin_centers - min(bin_centers) # scale values to interval [0,1]
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', C(c))
        ax.set_ylabel('Density')
        ax.set_xlabel('Colour (%s)' % cmap_name)
        plt.title('Colour Density over 1D Colour Axis')
        if logscale:
            plt.yscale('log', nonposy='clip')
        plt.margins(x=0)
        norm = None
        plt.register_cmap(cmap=C)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=C), ax=ax, orientation='horizontal', fraction=0.1)
        # Add KDE curve
        if add_kde_curve:
            kde = scipy.stats.kde.gaussian_kde(single_dim_P)
            kde_x = np.linspace(0, 1, kde_bins)
            ax.plot(kde_x, kde(kde_x), alpha=0.7, linestyle='dashed', linewidth=1.1, color='black')
    else:
        n, bins = np.histogram(single_dim_P, bins=nbins, range=(0,1), density=False)
    # Return: (1D pseudocolors per pixel, histogram values per bin, bin edge values, bin edge color values)
    return single_dim_P, n, bins, np.array([C(b)[0:3] for b in bins])

def display_hsv(img, orig_mask, args, fix_color_interval=True):
    names = ['Original', 'Hue', 'Saturation', 'Value']
    img = img[:,:,0:3] / 255

    if not orig_mask is None:
        print('\tApplying mask, but otherwise ignoring alpha channel')
        midvalue = 128
        bool_mask = (orig_mask > midvalue)
        f_mask = bool_mask.astype(int).astype(float) * (1.0 - 1e-6)
        img = img * f_mask[:,:,np.newaxis] #.reshape(img.shape[0], img.shape[1], 1)

    aspect_ratio = img.shape[0] / img.shape[1] #
    hsv_img = skcolor.rgb2hsv(img)
    fig, axs = plt.subplots(2, 2) #, figsize=(10, 3))
    c = 0
    q = []
    for i in range(2):
        for j in range(2):
            name = names[c]
            ax = axs[i,j]
            A = hsv_img[:,:,c - 1] if c > 0 else img # H, S, V
            cm_choice = 'hsv' if c == 1 else 'magma'
            # Note: nearest instead of bicubic avoids some odd interpolation
            # errors, especially for cyclic colormaps
            ims = ax.imshow(A, interpolation='nearest',
                            vmin=0, vmax=1, cmap=cm_choice)
            q.append(ims)
            ax.set_title(name.capitalize())
            plt.axis('off')
            ims.axes.get_xaxis().set_visible(False)
            ims.axes.get_yaxis().set_visible(False)
            c += 1
    # Neat: https://stackoverflow.com/questions/41428442/horizontal-colorbar-over-2-of-3-subplots
    #fig.colorbar(ims, ax=axs.ravel().tolist())
    # add_axes -> The dimensions [left, bottom, width, height] of the new axes.
    # subplots_adjust(left=None, bottom=None, right=None, top=None,
    #                 wspace=None, hspace=None)
    CW1 = 0.025
    CW2 = 0.04
    Rm = 0.87
    Bm = 0.13
    cb1_start = 0.55
    lstart = 0.125
    cb_pad = 0.02

    clen1 = 1.0 - cb1_start - 0.15
    clen2 = Rm - lstart
    fig.subplots_adjust(bottom=Bm, right=Rm, wspace=0.079, hspace=0.099)
    #fig.subplots_adjust(bottom=0.17, wspace=0.1, hspace=0.1)
    cbar_ax = fig.add_axes([Rm + cb_pad, cb1_start, CW1, clen1])
    fig.colorbar(q[1], cax=cbar_ax)
    cbar_ax = fig.add_axes([lstart, Bm - cb_pad, clen2, CW2])
    fig.colorbar(q[2], cax=cbar_ax, orientation='horizontal')

def plot_polar_generic(img, orig_mask, args, f, apply_mask=True, log_histo=False):
    """
    Args are the same as plot_polar_hsv except the colormap function
        f : [0,1] -> [0,1]^3
    is used to compute the scalar image used in or processed for the polar plot.
    """
    f = _to_cm_function(f)
    H, W, C = img.shape
    img = img[:,:,0:3] / 255 * 0.999
    img += 1e-5*np.random.rand(*img.shape)
    # RGB -> index value
    R, G, B = img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()
    tI = transform_rgb_to_cmap_index_vector(f, R, G, B, verbose=True).reshape(H,W)
    print("Index image shape", tI)
    # TODO
    if not orig_mask is None and apply_mask:
        midvalue = 128
        bool_mask = (orig_mask > midvalue)
        H = tI[bool_mask]
        print(H.shape, bool_mask.shape)
        rmnum = np.prod(tI.shape) - np.sum(bool_mask)
        print('\tRemoving', rmnum, "masked values (%.2f%%)"
              % (100 * rmnum / np.prod(tI.shape)))
    else:
        H = tI.flatten()
    _generate_polar_plot(H, f, log_histo=log_histo)

def _to_cm_function(f):
    # TODO option for manually defined mapping?
    if type(f) is str:
        return plt.cm.get_cmap(f)

def plot_colour_mapped_scalar_image(img, orig_mask, args, C):
    # TODO: option to pass C as a function
    Cname = C
    C = _to_cm_function(C)
    img = img[:,:,0:3] / 255
    H, W, _ = img.shape
    # RGB -> scalar index value
    R, G, B = img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()
    SI = transform_rgb_to_cmap_index_vector(C, R, G, B, verbose=True).reshape(H, W)
    #
    fig, ax = plt.subplots()
    ims = ax.imshow(SI, interpolation='nearest',
                    vmin=0, vmax=1, cmap=Cname)
    fig.colorbar(ims, #plt.cm.ScalarMappable(norm=norm, cmap=C),
                 ax=ax, orientation='vertical', fraction=0.1)

def plot_polar_hsv(img, orig_mask, args, apply_mask=True, log_histo=False):
    print('Generating polar plot')
    img = img[:,:,0:3] / 255
    hsv_img = skcolor.rgb2hsv(img)
    if not orig_mask is None and apply_mask:
        midvalue = 128
        bool_mask = (orig_mask > midvalue)
        H = hsv_img[:,:,0][bool_mask]
        print(H.shape, bool_mask.shape)
        rmnum = np.prod(hsv_img[:,:,0].shape) - np.sum(bool_mask)
        print('\tRemoving', rmnum,
              "masked values (%.2f%%)"
              % (100 * rmnum / np.prod(hsv_img[:,:,0].shape)))
    else:
        H = hsv_img[:,:,0].flatten()
    xticklabels = [ 'Red', 'Orange', 'L. Green', 'Green',
                    'L. Blue', 'Blue', 'Purple', 'Pink'   ]
    _generate_polar_plot(H, plt.cm.hsv, xlabels=xticklabels, log_histo=log_histo)

def _generate_polar_plot(H, ff, xlabels=None, log_histo=False, n_bins=20):
    print("H min/max/mean/samples\n",
        H.min(), H.max(), H.mean(), H[np.random.randint(0,H.shape[0],50)] )
    H = 2 * np.pi * H
    m = H.shape[0]
    print('Num values in histogram:', m)
    fig = plt.figure()
    #-----------------#
    N = n_bins # N_bins
    bottom = 0.0
    max_height = 1.0
    #-----------------#
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    # hist : array -> The values of the histogram. (counts)
    # bin_edges : array of dtype float -> Return the bin edges (length(hist)+1).
    hist, bin_edges = np.histogram(H, bins = N, range=(0.0, 2 * np.pi))
    if log_histo:
        hist = np.log(hist + 1.0)
    print('hist', hist)

    radii = max_height * hist / m
    width = (2 * np.pi) / N

    ax = plt.subplot(111, polar=True)
    ax.set_yticklabels([])
    if xlabels is None:
        xlabels = [""] * 8
    ax.set_xticklabels(xlabels)
    centers = [ 'left', 'left', 'center', 'right',
                'right', 'right', 'center', 'left' ]
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_ha(centers[i])

    bars = ax.bar(theta, radii, width=width, bottom=bottom)

    # Use custom colors and opacity
    assert len(radii) == N
    for i, (r, bar) in enumerate( zip(radii, bars) ):
        curr_val = i / N
        bar.set_facecolor( ff(curr_val) )
        bar.set_alpha(0.99)

def transform_rgb_to_cmap_index_vector(C, R, G, B, verbose=True, n_search_bins=200):
    """ Takes normed image values """
    if type(C) is str: C = plt.cm.get_cmap(C)
    # Use the colormap to get the bin values
    bin_edge_indices = np.linspace(0.0, 1.0, n_search_bins)
    bin_middle_inds = (bin_edge_indices[1:] + bin_edge_indices[:-1]) / 2.0
    bin_middle_colors = np.array([ C(midind)[0:3] for midind in bin_middle_inds ]) # remove alpha after mapping
    if verbose: print(bin_middle_inds, '\n', bin_middle_colors, '\n', bin_middle_inds.shape, bin_middle_colors.shape)
    # Put image pixels into normalized 3D RGB colour space
    P = np.vstack( (R, G, B) ).T # / 255.0
    # We want to map each 3D colour to its 1D representation to get a histogram of it
    # Let's get the nearest neighbour of each p in P, from within the set of mapped
    # middle bin colors with a KD-tree
    from scipy import spatial
    if verbose: print('Building and querying KD-tree')
    tree = spatial.cKDTree(bin_middle_colors)
    distances, neb_indices = tree.query(P, k=1)
    if verbose: print('Distance (error) max and avg', np.mean(distances), np.max(distances))
    # Assign 1D pseudo-colour values to each position in the colour point cloud of the image
    single_dim_P = bin_middle_inds[ neb_indices ]
    if verbose:
        rinds = np.random.randint(0, single_dim_P.shape[0], 15)
        print('Single dim values obtained', single_dim_P[rinds])
        print('RGBs', R[rinds], G[rinds], B[rinds])
    return single_dim_P

def display_orig(img, orig_mask, args):
    imdisplay(img, title="Original Image (%s)" % args.input)
    if not orig_mask is None:
        fig, ax = plt.subplots()
        ax.imshow(orig_mask)
        plt.title('Mask (%s)' % args.input)

def plot_density_proj(R, G, B,
                      verbose=True,
                      point_subsample=1000,
                      density_subsample=20000,
                      nbins=30):
    print('Generating 3D scatterplot with projected densities')
    n_pixels = len(R)
    if verbose: print("n_pixels", n_pixels)
    origR, origG, origB = R, G, B
    orig_positions = np.vstack( (origR, origG, origB) ).T
    if not point_subsample is None and n_pixels > point_subsample:
        if verbose: print("Subsampling pixels for display")
        rinds = np.random.choice(n_pixels, size=point_subsample, replace=False)
        R, G, B = R[rinds], G[rinds], B[rinds]
    positions = np.vstack( (R, G, B) ).T
    if verbose: print("pos shape", positions.shape)
    colours = positions / 255
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(R, G, B, s=3, c=colours, alpha=1)
    ax.set_xlabel('Red'); ax.set_ylabel('Green'); ax.set_zlabel('Blue')
    # Subsampling for KDE
    if n_pixels > density_subsample:
        if verbose: print('Subsampling for density calculation')
        rindsd = np.random.choice(n_pixels, size=density_subsample, replace=False)
        new_positions = orig_positions[rindsd, :]
    else:
        new_positions = orig_positions
    RG = new_positions[:, (0,1)]
    RB = new_positions[:, (0,2)]
    GB = new_positions[:, (1,2)]
    if verbose: print("2d shapes", RG.shape, RB.shape, GB.shape)
    from scipy.stats import kde
    def generate_contour_plot(ind):
        if verbose: print('Computing density', ind)
        data = RG if ind == 0 else (RB if ind == 1 else GB)
        kde = scipy.stats.kde.gaussian_kde(data.T)
        eval_x, eval_y = np.mgrid[0:255:nbins*1j, 0:255:nbins*1j]
        kde_out = kde(np.vstack([eval_x.flatten(), eval_y.flatten()]))
        return eval_x, eval_y, kde_out.reshape(eval_x.shape)
    RG_vals, RB_vals, GB_vals = [generate_contour_plot(ii) for ii in range(3)]
    if verbose: print('kde shape', RG_vals[0].shape, RG_vals[1].shape, RG_vals[2].shape)
    # Settings for KDE plots
    cmap = plt.cm.hot_r #plt.cm.nipy_spectral_r
    max_alpha = 0.9
    # Manually define alpha
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, max_alpha, cmap.N)
    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(my_cmap)
    cstr, rstr = 1, 1
    # Add projected plots to the walls in 3D
    def add_imgs(xx,yy,zz,H):
        norm = matplotlib.colors.Normalize(vmin=H.min(), vmax=H.max() )
        colors = my_cmap(norm(H))
        ax.plot_surface(xx, yy, zz, cstride=cstr, rstride=rstr, facecolors=colors, shade=False,
                        #alpha=img_alpha,
                        linewidth=0, antialiased=False)
    add_imgs(RG_vals[0], RG_vals[1], np.zeros_like(RG_vals[0]), RG_vals[2])
    add_imgs(RB_vals[0], np.zeros_like(RB_vals[0]) + 255, RB_vals[1], RB_vals[2])
    add_imgs(np.zeros_like(GB_vals[0]), GB_vals[0], GB_vals[1], GB_vals[2])

def plot_3D_rgb(R, G, B, nbins=8):
    """
    Displays a 3D interactive histogram of the pixel colors.
    """
    print("Generating 3D RGB histogram")
    rngs= [(0, 255) for _ in range(3)]
    h, e = np.histogramdd([R, G, B], bins=nbins, range=rngs, density=True)
    from utils import histogram3dplot as histo3d
    fig = plt.figure(figsize=(8,8))
    histo3d(h, e, fig=fig)

def plot_1D_rgb(R,G,B,nbins=40):
    """
    Displays a plot with the 1D pixel value distributions per channel.
    """
    print("Generating 1D RGB histograms")
    cm_names = ("Reds", "Greens", "Blues")
    title_names = ["Histogram of %s Pixel Values" % s for s in ("Red", "Green", "Blue") ]
    # https://stackoverflow.com/questions/37360568/python-organisation-of-3-subplots-with-matplotlib
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(15,5))
    def gen_subplot(pos):
        V = R if pos == 0 else (G if pos == 1 else B)
        ax1 = fig.add_subplot(gs[0,pos])
        cm1 = plt.cm.get_cmap(cm_names[pos])
        n, bins, patches = ax1.hist(V, bins=nbins, density=1, range=(0,255),
                                    linewidth=1.2, edgecolor='black') # color overridden
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers) # scale values to interval [0,1]
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm1(c))
        plt.title(title_names[pos])
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    gen_subplot(0) # Red
    gen_subplot(1) # Green
    gen_subplot(2) # Blue

if __name__ == '__main__':
    overall_desc = 'Implements several functions useful for visualization of images, particularly color distributions.'
    parser = argparse.ArgumentParser(description=overall_desc)
    # Input image
    parser.add_argument('input', type=str, help='Input: either a folder or an image')
    # Preprocessing options
    parser.add_argument('--blur', type=float, default=0.0,
        help='Specify Gaussian blur standard deviation applied to the image')
    # Plotting options
    parser.add_argument('--show_img', dest='show_img', action='store_true',
        help='Whether to display the input image')
    parser.add_argument('--show_hsv', dest='show_hsv', action='store_true',
        help='Whether to display the input image as HSV components')
    parser.add_argument('--hist_hsv_polar', dest='hist_hsv_polar', action='store_true',
        help='Displays a polar (circular) plot of the HSV hue values')
    parser.add_argument('--show_twilight_img', dest='show_twilight_img', action='store_true',
        help='Whether to display the input image based on its scalar twilight colormap values')
    parser.add_argument('--hist_twilight_polar', dest='hist_twilight_polar', action='store_true',
        help='Displays a polar (circular) plot of the color values from the cyclic twilight map')
    parser.add_argument('--hist_rgb_1d', dest='hist_rgb_1d', action='store_true',
        help='Displays a set of 1D histograms of RGB pixel values')
    parser.add_argument('--hist_rgb_3d', dest='hist_rgb_3d', action='store_true',
        help='Displays a 3D histogram of RGB pixel values')
    parser.add_argument('--scatter_densities', dest='scatter_densities', action='store_true',
        help='Displays a 3D histogram of RGB pixel values')
    parser.add_argument('--manual_unfolded_1d', dest='manual_unfolded_1d', action='store_true',
        help='Displays a curve of the RGB values unfolded onto a 1D axis')
    parser.add_argument('--projected_pixels', dest='projected_pixels', action='store_true',
        help='Displays a projection of the RGB pixel values into a data-dependent subspace')
    # Writing options
    parser.add_argument('--write_1d_histo_vals', dest='write_1d_histo_vals', action='store_true',
        help='Writes the 1D unfolded colour histogram values of each image to a CSV')
    parser.add_argument('--output_file', dest='output_file', type=str, default='histogram-1D_unfolded.csv',
        help='File into which to write the histogram values (default: "histogram-1D_unfolded.csv")')
    # Meta arguments
    parser.add_argument('--all', dest='all', action='store_true',
        help='If given, all visualizations will be displayed')
    args = parser.parse_args()
    # Special case: if desired, write out 1D histogram values per image in a folder
    if args.write_1d_histo_vals:
        main_dir_write(args)
        sys.exit(0)
    # Otherwise, it must be a single image, which we visualize
    if args.all:
        args.show_img = True
        args.hist_rgb_1d = True
        args.show_hsv = True
        args.hist_hsv_polar = True
        args.hist_rgb_3d = True
        args.scatter_densities = True
        args.manual_unfolded_1d = True
        args.projected_pixels = True
        args.hist_twilight_polar = True
        args.show_twilight_img = True
    if args.hist_hsv_polar:
        args.show_hsv = True
    if args.hist_twilight_polar:
        args.show_twilight_img = True
    main_single_display(args)

# TODO: 2D densities with marginal curves



#
