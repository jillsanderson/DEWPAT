import os, sys, numpy as np, skimage, argparse, warnings, matplotlib, csv
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D, axes3d
from skimage import io
from .prob import gaussian_prob_divergence

### Visualization helpers ###

# Helper function for displays
def imdisplay(inim, title, colorbar=False, cmap=None, mask=None, vmin=None, vmax=None, 
              keep_axis_info=False, suptitle=None):
    """ Displays the input image 'inim'.

    Args:
        inim: The input image.
        title: The title of the plot.
        colorbar: Optional; Add a colorbar to the plot if True.
          False by default.
        cmap: Optional; The colormap used to map scalar values to
          3D pixel values. Only used if 'inim' is a greyscale image.
          Can be a matplotlib.colors.ColorMap instance or the name of
          a color map. None by default.
          See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        mask: Optional; The binary alpha mask of the image.
        vmin: Optional; The minimum scalar value covered by colormap 'cmap'.
          None by default.
        vmax: Optional; The maximum scalar value covered by colormap 'cmap'.
          None by default.
    """
    fig, ax = plt.subplots()
    if (not mask is None) and (not type(mask) is bool):
        if type(cmap) is str: 
            cmap = matplotlib.cm.get_cmap(name=cmap)
        cmap.set_bad(color='white')
        inim = inim.copy()
        inim[ mask == 0 ] = np.nan
        #print('Setting masked pixels to mapped NaNs')
    imm = ( ax.imshow(inim, vmin=vmin, vmax=vmax) 
            if cmap is None else 
            ax.imshow(inim, cmap=cmap, vmin=vmin, vmax=vmax) )
    plt.title(title)
    fig.canvas.set_window_title(title)
    if not suptitle is None: plt.suptitle(suptitle)
    if colorbar: fig.colorbar(imm)
    if not keep_axis_info:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

def patch_display(patches, nrows, ncols, show=False, title=None, subtitles=None, hide_axes=False):
    """ Displays a set of patches on the same plot.

    The number of patches in 'patches' must be equal to
    'nrows' * 'ncols'.

    Args:
        patches: An array of N patches.
        nrows: The number of rows.
        ncols: The number of columns.
        show: Optional; Immediately displays the plot with a
          if True. If set to False (default), the plot can be
          displayed after the function ends with a call to 
          matplotlib.pyplot.show(). 
        title: Optional; The title of the window displaying
          the patches. None by default.
        subtitles: Optional; An array of N strings used to
          label the patches. If set to None (default), the patches
          will be labelled with its row and column number on the plot.
        hide_axes: Optional; Displays the axes labels on each
          patch if True. False by default.
    """
    # patches must be n_patches x H x W x C
    N = nrows * ncols
    N_dims = len(patches.shape)
    assert N == patches.shape[0]
    if not subtitles is None: assert len(subtitles) == N
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        #img = np.random.randint(10, size=(h,w))
        if N_dims == 4:
            axi.imshow(patches[i,:,:,:])
        else: # Mask case
            axi.imshow(patches[i,:,:], vmin=0, vmax=1)
        #axi.imshow(img, alpha=0.25)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        if subtitles is None:
        # write row/col indices as axes' title for identification
            axi.set_title("R"+str(rowid)+"C"+str(colid))
        else:
            subtitle = subtitles[i]
            axi.set_title( str(subtitle) )
        if hide_axes:
            axi.get_xaxis().set_visible(False)
            axi.get_yaxis().set_visible(False)
            #plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.tight_layout(pad=1.5, w_pad=0.1, h_pad=1.0)
    if not title is None: 
        fig.canvas.set_window_title(title)
        fig.suptitle(title)
        fig.subplots_adjust(top = 0.87)
    if show: plt.show()

def histogram3dplot(h, e, fig=None, verbose=True):
    """ Visualize a 3D histogram

    Adapted from:
        https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/IP/Images/ImageHistograms.html

    Args:
        h: histogram array of shape (M, N, O)
        e: list of bin edge arrays (for R, G and B)
        fig: Optional; A matplotlib.pyplot.figure
          instance to display the plot. If None (default),
          the current figure (or a new figure if one does
          not exist) will be used.
        verbose: Optional; Print verbosely if True. True
          by default.
    """
    M, N, O = h.shape
    idxR = np.arange(M)
    idxG = np.arange(N)
    idxB = np.arange(O)

    R, G, B = np.meshgrid(idxR, idxG, idxB)
    a = np.diff(e[0])[0]
    b = a/2
    R = a * R + b

    a = np.diff(e[1])[0]
    b = a/2
    G = a * G + b

    a = np.diff(e[2])[0]
    b = a/2
    B = a * B + b

    cord = 'C'
    flat_R, flat_G, flat_B = (R.flatten(order=cord), G.flatten(order=cord), B.flatten(order=cord))

    if verbose: print('r,g,b shapes', R.shape, G.shape, B.shape)

    h = h / np.sum(h)
    flat_h = h.flatten(order='F')

    if verbose: print('h shape', h.shape)

    if fig is not None:
        f = fig
        #f = plt.figure(fig)
    else:
        f = plt.gcf()
    ax = f.add_subplot(111, projection='3d')
    mxbins = np.array([M,N,O]).max()
    if verbose: print('mxbins', mxbins)

    # Note: somehow the values have been permuted. I suspect it is due to the flattening
    # procedures. This appears to give correct results, nevertheless.
    # The permuting has caused R->B, B->G, G->R; hence we cycle their order.

    colors = np.vstack( (flat_B, flat_R, flat_G) ).T / 255
    if verbose: print('color shape', colors.shape)

    ax.scatter(flat_B, flat_R, flat_G, s=flat_h*(256/mxbins)**3/2, c=colors)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

### Patch extraction helpers ###

def patches_over_channels(img, patch_size, window_step, return_meta=True, floatify=False):
    """ Generates square patches from a 3-channel image.

    The patches will be overlapping if 'window_step' is smaller than
    any of the 'patch_size' dimensions.

    Args:
        img: A 3-channel image.
        patch_size: The length of the square patch in pixels.
        window_step: The step size in pixels between adjacent patches.
        return_meta: Optional; If True, return meta data in addition to
          the patches (see below).
        floatify: Optional; If True, patches will be returned in floating
          point format.
        
    Returns:
        An array of length 3 (one per channel) where each element is a matrix
        of 2D patches for the corresponding greyscale image.
        
        If 'return_meta' is True, also return:
          - The shape of the array (3, H, W, Px, Py) where, H is the number
              of vertical patches, W is the number of horizontal patches, and Px x Py
              is the dimensions of the square patch.
          - The area of a patch (Px x Py).
    """
    # C x H x W x Px x Py = num_channels x n_patches_vert x n_patches_horz x patch_H x patch_W
    if floatify:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = skimage.img_as_float(img)
    P = np.array([
            patches_per_channel(img[:,:,k], patch_size, window_step)
            for k in range(3)
        ])
    if return_meta: return P, P.shape, patch_size**2
    return P

def patches_per_channel(channel, patch_size, window_step):
    """ Generates square patches from a greyscale image.

    The patches will be overlapping if 'window_step' is smaller than
    any of the 'patch_size' dimensions.

    Args:
        channel: A greyscale image.
        patch_size: The length of the square patch size in
          pixels.
        window_step: The step size in pixels between adjacent patches.

    Returns:
        An H x W matrix of 2D patches where H is the number of vertical
        patches and W is the number of horizontal patches.
    """
    # H x W x Px x Py = n_patches_vert x n_patches_horz x patch_H x patch_W
    return view_as_windows(
                np.ascontiguousarray( channel ),
                (patch_size, patch_size),
                step=window_step)

def vectorize_single_masked_patch(patches, mask, i, j, wt):
    """ Vertorize patch from 'patches' at row 'i' and column 'j'.

    None is returned if any pixel in the corresponding mask
    contains a 0.

    Args:
        patches: An array of patches (see patches_over_channels).
        mask: A matrix of patches corresponding to the alpha masks
          of 'patches'.
        i: The vertical index of the patch.
        j: The horizontal index of the patch.
        wt: The area of the patch.
    
    Returns:
        A ('wt' x 3) length vector of the patch values. If any
        pixel in the patch's mask is 0, None is returned instead.
    """
    mask_patch = mask[i,j,:,:]
    if 0 in mask_patch: return None
    unfolded_patch = patches[:,i,j,:,:].reshape(wt * 3)
    return unfolded_patch

def vectorize_single_masked_patch_as_list(patches, mask, i, j, wt):
    """ Vertorize patch from 'patches' at row 'i' and column 'j'.

    None is returned if any pixel in the corresponding mask
    contains a 0.

    Args:
        patches: An array of patches (see patches_over_channels).
        mask: A matrix of patches corresponding to the alpha masks
          of 'patches'.
        i: The vertical index of the patch.
        j: The horizontal index of the patch.
        wt: The area of the patch.
    
    Returns:
        A 'wt'-vector of 3D pixels from the patch. If any pixel in
        the patch's mask is 0, None is returned instead.
    """
    mask_patch = mask[i,j,:,:]
    if 0 in mask_patch: return None
    listified_patch = patches[:,i,j,:,:].reshape(3, wt).T
    return listified_patch

def vectorize_masked_patches(patches, mask, H, W, as_list=False, flatten=True, remove_none=True):
    """ Vectorizes patches in 'patches'.

    Patches with a mask containing a 0 will appear as None in the
    result unless 'remove_none' is True.

    Args:
        patches: An array of patches (see patches_over_channels).
        mask: A 2D matrix of patches corresponding to the
          alpha masks of 'patches'.
        H: The number of vertical patches in 'patches'.
        W: The number of horizontal patches in 'patches'.
        as_list: Optional; If True, the vector will contain RGB
          pixels from the patches. Otherwise, the pixels will be
          flattened in the vector. False by default.
        flatten: Optional; If True (default), the return value will
          be a list of patches of length H x W. Otherwise, the
          return value will be an H x W matrix of patches.
        remove_none: Optional; If True (default), patches with masked
          pixels (which appear as None) will be removed from the return
          value. Note that the result will be a 1D list of vectors even
          if flatten is False.

    Returns:
        A list/matrix of patch vectors.
    """
    wt = mask.shape[2] * mask.shape[3] # patch/window total size
    if flatten:
        if as_list:
            P = [ vectorize_single_masked_patch_as_list(patches, mask, i, j, wt)
                  for i in range(H) for j in range(W) ]
        else:
            P = [ vectorize_single_masked_patch(patches, mask, i, j, wt)
                  for i in range(H) for j in range(W) ]
    else:
        if as_list:
            P = [ [ vectorize_single_masked_patch_as_list(patches, mask, i, j, wt)
                    for i in range(H) ] for j in range(W) ]
        else:
            P = [ [ vectorize_single_masked_patch(patches, mask, i, j, wt)
                    for i in range(H) ] for j in range(W) ]
    if remove_none:
        return np.array([vp for vp in P if not vp is None])
    else:
        return P

### Image processing helpers ###

def generate_gradient_magnitude_image(img, divider=2, to_ubyte=False):
    """ Generates the per-channel L2 gradient magnitude image of the input `img`.

    The gradient is estimated using the Scharr transform.

    Args:
        img: The input image.
        divider: Optional; Each edge pixel value is divided by `divider` (enforce
          output in [0,1] + numerical stability). 2 by default.
        to_ubyte: Optional; If True, the output is converted to the ubyte numpy dtype.
          False by default.

    Returns:
        The gradient magnitude image.
    """
    @adapt_rgb(each_channel)
    def cgrad_x(img):
        return filters.scharr_v(img)
    @adapt_rgb(each_channel)
    def cgrad_y(img):
        return filters.scharr_h(img)
    Ig_x, Ig_y = cgrad_x(img), cgrad_y(img)
    Ig_x_sq, Ig_y_sq = Ig_x * Ig_x, Ig_y * Ig_y
    gradient_img = np.sqrt(Ig_x_sq + Ig_y_sq) / divider # to remain in [0,1]
    # print('GradImg min/max: %.2f/%.2f' % (gradient_img.min(), gradient_img.max()))
    if to_ubyte:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gradient_img = skimage.img_as_ubyte(gradient_img)
    return gradient_img

def gaussian_blur(image, sigma):
    """ Performs Gaussian blurring on 'image' with standard deviation 'sigma'.

    Args:
        image: The input image.
        sigma: The standard deviation of Gaussian kernel.

    Returns:
        The blurred image.
    """
    return np.clip(filters.gaussian(image, sigma=sigma, multichannel=True), 0.0, 1.0)

def to_perceptual_greyscale(img):
    """ Performs perceptual luminance-preserving RGB to greyscale conversion.

    The result is in ubyte format (pixel values in [0, 255]).

    Args:
        img: The RGB image.
    
    Returns:
        The greyscale image.
    """
    from skimage.color import rgb2gray
    new_img = dupe_channel( rgb2gray(img) )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_img = skimage.img_as_ubyte( new_img )
    return new_img

def to_avg_greyscale(img):
    """ Converts an image to greyscale by averaging the channel pixel values.

    The result is in ubyte format (pixel values in [0, 255]).

    Args:
        img: The RGB image.
    
    Returns:
        The greyscale image.
    """
    new_img = dupe_channel( np.clip(np.mean(img / 255, axis=-1), 0, 1) )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_img = skimage.img_as_ubyte( new_img )
    return new_img

def conv_to_ubyte(img):
    """ Converts an image to unsigned byte format [0, 255].

    See skimage.img_as_ubyte().

    Args:
        img: The RGB image.
    
    Returns:
        The ubyte image.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_ubyte( img )
    return img

### Timing helpers ###

# Note: calling this as a decorator actually runs it, so that
# replacer is the actual decorator used.
def timing_decorator():
    """ A decorator that returns the timing of the wrapped function.
    """
    def decorator(wrapped_func):
        def replacement(*args, **kwargs):
            start = timer()
            output = wrapped_func(*args, **kwargs)
            end = timer()
            if 'timing' in kwargs and not kwargs['timing']:
                return output
            return output, (end - start)
        return replacement
    return decorator

### Array Helpers ###

def dupe_channel(c):
    """ Duplicates a single channel to form a 3-channel image.

    Args:
        c: A greyscale image.
    
    Returns:
        A 3-channel image.
    """
    return combine_channels([c, c, c])

def combine_channels(channels):
    """ Combines channels into a multi-channel image.

    Args:
        channels: A list of greyscale images.
    
    Returns:
        A multi-channel image.
    """
    return np.stack( channels, axis = -1 )

### Colour Helpers ###

def color_multiinterpolator(color_list, times=None):
    """ Produces a color function f : [0,1] -> R^3 that linearly interpolates over 'color_list'.

    Args:
        color_list: A list of color arrays, with length n.
        times: Optional; Provides a mapping of values to the corresponding colors in 'color_list'.
            Must be of length n-2 and each entry has 0 < t_i < 1 (the first and last colors are 
            mapped to by 0 and 1 respectively). If set to None (default), the colors are equidistantly
            mapped from [0, 1] to the colors in 'color_list'.

    Returns:
        A matplotlib.colors.LinearSegmentedColormap instance.
    """
    n = len(color_list)
    if n == 2: return Color.color_biinterpolator(color_list[0], color_list[1])
    if times is None: times = np.linspace(0.0, 1.0, num=n)[1:-1]
    if not times is None:
        assert len(times) + 2 == n, 'Times length mismatch'
        assert all( (t >= 0.0 and t <= 1.0) for t in times ), 'Untenable times choices'
        times = [0.0] + list(times) + [1.0]
    # Now generate the interpolator function
    col_list = [ (t, c) for c, t in zip(color_list, times) ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("wtf", col_list)

def color_biinterpolator(c1, c2):
        return lambda t: c2*t + (1-t)*c1

def from_ints(r,g,b):
    return np.array([r, g, b, 255]) / 255.0


def plotDimensionallyReducedVectorsIn2D(vectors, method='pca', manifold_learning_options={},
                                        point_labels=None, colors=None, verbose=True):
    """ Dimensionally reduces a set of n-dimensional vectors to 2D.

    The reduced vectors are plotted. If 'point_labels' is provided, 'colors' is ignored.

    Args:
        vectors: An m x n matrix.
        method: optional; The method used to reduce the vectors. Must be
          one of the following:
            'pca'     - Principal Component Analysis (default)
            'tsne'    - t-distributed Stochastic Neighbor Embedding
            'isomap'  - Isomap Embedding
            'lle'     - Locally Linear Embedding
        manifold_learning_options: optional; A dict containing learning
            options. The options allowed depend on the method used:
            'pca'     - 'whiten'             Perform PCA Whitening if True. False by
                                             default
            'tsne'    - 'learning_rate'      The learning rate. Recommended to be
                                             in the range [10, 1000]. 200.0 by default.
                      - 'perplexity'         An estimate of the number of close
                                             neighbors around each point. Recommended
                                             to be in the range [5, 50]. 30.0 by default.
                      - 'early_exaggeration' Controls how tight natural clusters in the original
                                             space are in the embedded space and how much space 
                                             will be between them. 12.0 by default.
            'isomap'  - 'n_neighbors'        The number of neighbors to consider for each point.
                                             5 by default.
            'lle'     - 'n_neighbors'        The number of neighbors to consider for each point.
                                             5 by default.
        point_labels: optional; A list of m ints/floats used to color the points on the plot.
          The labels are mapped onto the pyplot.cm.jet colormap. Can only be used if m <= 256.
          None by default.
        colors: optional; A m x 3 or m x 4 matrix of RGB/RGBA pixels used to color the points
          on the plot. None by default.
        verbose: optional; Print verbosely if True. True by default.

    Returns:
        An m x 2 matrix of the reduced vectors.
    """
    method = method.lower()
    methods = ['pca', 'tsne', 'isomap', 'lle']
    assert method in methods, 'Method must be one of' + str(methods)
    if not point_labels is None:
        assert len(point_labels) == len(vectors), "Vectors-to-labels mismatch"
        # assert all([type(pl) is int for pl in point_labels]), "Point labels must be integers"

    # Create figure
    f = plt.figure()
    a = f.gca()

    ### Manifold Learning / Dimensionality Reduction ###
    # At the end, we get x_new as an m x 2 matrix
    val_with_default = lambda val, default: val if not val is None else default
    if method == 'pca':
        if verbose: print('Applying PCA')
        from sklearn.decomposition import PCA
        whiten = val_with_default( manifold_learning_options.get('whiten'), False )
        pca = PCA(n_components=2, whiten=whiten)
        x_new = pca.fit_transform(vectors)
    elif method == 'tsne':
        if verbose: print('Applying T-SNE')
        from sklearn.manifold import TSNE
        learning_rate = val_with_default(
                            manifold_learning_options.get('learning_rate'), 200.0 )
        perplexity = val_with_default(
                            manifold_learning_options.get('perplexity'), 30.0 )
        early_exaggeration = val_with_default(
                            manifold_learning_options.get('early_exaggeration'), 12.0 )
        x_new = TSNE(n_components = 2,
                     perplexity = perplexity,
                     early_exaggeration = early_exaggeration,
                     learning_rate = learning_rate
                ).fit_transform(vectors)
    elif method == 'isomap':
        if verbose: print('Applying Isomap')
        from sklearn.manifold import Isomap
        n_neighbors = val_with_default(manifold_learning_options.get('n_neighbors'), 5)
        x_new = Isomap(n_neighbors=n_neighbors).fit_transform(vectors)
    elif method == 'lle':
        if verbose: print('Applying LLE')
        from sklearn.manifold import LocallyLinearEmbedding
        n_neighbors = val_with_default(manifold_learning_options.get('n_neighbors'), 5)
        x_new = LocallyLinearEmbedding(n_neighbors=n_neighbors).fit_transform(vectors)

    # Unzip the data matrix
    x, y = x_new[:,0], x_new[:,1]

    ### Add labels to the plot if given ###
    if not point_labels is None:
        unique_labels = list(set(point_labels))
        N = len(unique_labels)
        # If labels are not integers, make them so
        if not type(point_labels[0]) is int:
            label_dict = { unlab : i for i, unlab in enumerate(unique_labels) }
            point_labels = [ label_dict[p_lab] for p_lab in point_labels ]
        # Adapted from the method given here:
        # https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
        # Define the colormap
        cmap = plt.cm.jet
        # Extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # Create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # Define the bins and normalize
        bounds = np.linspace(0, N, N + 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # Create figure
        splot = a.scatter(x, y,
                          c=point_labels,
                          cmap=cmap,
                          norm=norm)
    else:
        if colors is None:
            splot = a.scatter(x, y)
        else:
            splot = a.scatter(x, y, c=colors, alpha=0.7, s=4)
    a.set_ylabel('Coordinate 2')
    a.set_xlabel('Coordinate 1')
    plt.title('Projected Pixels in Subspace')

### IO ###

def load_helper(image_path, blur_sigma=None, apply_alpha_to_rgb_channels=True, verbose=True):
    """ Read the image at the specified file path.

    Args:
        image_path: The path of the image.
        blur_sigma: Optional; The standard deviation of the Gaussian
          kernel used to blur the image. Must be >= 0. Set to None
          (default) to skip this step.
        apply_alpha_to_rgb_channels: Optional; If True, any pixel with
          an alpha value <= 128 will be removed from the RGB vectors in
          the return value.
        verbose: Optional; Print verbosely if True. True by default.

    Returns:
        A tuple containing:
            1) The image object (H x W x 3)
            2) A vector of red pixel values from the image
            3) A vector of green pixel values from the image
            4) A vector of blue pixel values from the image
            5) A binary alpha mask (H x W) containing 1 for each
               pixel with an alpha value > 128 and 0 otherwise.
               If the image has no alpha channel, None is returned
               instead.
    """
    img = io.imread(image_path)
    if not blur_sigma is None:
        assert blur_sigma >= 0.0, "Untenable blur kernel width"
        if blur_sigma > 1e-5:
            if verbose: print("\tBlurring with sigma =", blur_sigma)
            #bb = gaussian_blur(img, blur_sigma)
            #print(bb.max(), bb.min())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = skimage.img_as_ubyte( gaussian_blur(img, blur_sigma) )
    H, W, C = img.shape
    if verbose: print("Loaded image", image_path, "(Size: %s)" % str(img.shape))
    # Computing mask
    alpha_present = (C == 4)
    if alpha_present:
        if verbose: print("\tAlpha channel present. Generating mask.")
        midvalue = 128
        orig_mask = img[:,:,3]
        bool_mask = (orig_mask > midvalue)
        int_mask = bool_mask.astype(int)
        img = img[:,:,0:3]
        if apply_alpha_to_rgb_channels:
            img = int_mask.reshape(H,W,1) * img
        R, G, B = (img[:,:,0][bool_mask], img[:,:,1][bool_mask], img[:,:,2][bool_mask])
    else:
        if verbose: print("\tNo alpha channel present.")
        R, G, B = img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()
        orig_mask = None
    # At this point, img is always H X W x 3; mask is either None or H x W (boolean/byte)
    # R, G, and B are vectors; img is ubyte type.
    return img, R, G, B, orig_mask

### Block extraction and pairwise moment comparison methods ###

def pairwise_moment_distances(img, mask, block_cuts, gamma_mu_weight, gamma_cov_weight, display_intermeds, verbose, mode='central'):
    """ Computes the pairwise moment distance between non-overlapping patches.

    The input image is divided into non-overlapping patches. The measure
    depends on the mode selected (see below).
    Note: if a patch is *entirely* masked, it is removed from consideration.

    Args:
        img: The input image.
        mask: The alpha mask of the image.
        block_cuts: A list [x, y] representing the height and width
          of the patches. If the height or width of 'img' is not a
          multiple of x or y, the image is shaved from the bottom
          and/or left.
        gamma_mu_weight: The weight given to the mean distance
          between patches. If 'mode' is not 'central' this must be
          set to None.
        gamma_cov_weight: The weight given to the covariance distance
          between patches. If 'mode' is not 'Central' this must be
          set to None.
        display_intermeds: If True, the image and mask patches will
          be displayed.
        verbose: If True, print verbosely.
        mode: Optional; Must be one of the following options:

          [ 'central', 'pw_symmetric_KL',
            'pw_W2', 'pw_Hellinger',
            'pw_bhattacharyya', 'pw_FMATF' ]

            The default mode 'central' considers the first (mean) and
            second (covariance) moments across patches. The remaining
            modes compute the distributional divergences between patches,
            based on an underlying Gaussian prior assumption, using the
            specified method.
    
    Returns:
        The mean pairwise moment distance.

    TODO: options for other matrix norms in central moment distance.
    """
    assert img.max() > 1, "Unexpected pixel scaling encountered"
    if mode != 'central': 
        assert gamma_mu_weight is None and gamma_cov_weight is None
        assert mode in [ 'pw_symmetric_KL',  'pw_W2', 'pw_Hellinger',
                         'pw_bhattacharyya', 'pw_FMATF' ]
    # Correct the image pixel values
    img = img / 255.0
    ### Extract non-overlapping image blocks ###
    img_blocks, mask_blocks = blockify(img, mask, block_cuts, verbose)
    if verbose: print("Blockified sizes", img_blocks.shape, mask_blocks.shape)
    NBs_h, NBs_w, BS_h, BS_w, C = img_blocks.shape
    nr, nc, N_total = NBs_h, NBs_w, NBs_h * NBs_w
    patches = img_blocks.reshape(N_total, BS_h, BS_w, C)
    mpatches = mask_blocks.reshape(N_total, BS_h, BS_w)
    if display_intermeds:
        patch_display(patches, nr, nc, show=False, 
                title="Image Patches", subtitles=None, hide_axes=True)
        patch_display(mpatches, nr, nc, show=False, 
                title="Mask Patches",  subtitles=None, hide_axes=True)
    ### Now compute the pairwise distances between the moments ###
    # Using masked arrays to get moments for masked patches
    def get_masked_array(patch, patch_mask):
        unrolled = patch.reshape(-1, 3)
        _um = patch_mask.reshape(unrolled.shape[0])
        # In numpy, we mask/ignore values where the mask value is True (rather than False)
        # So notice we *invert* the mask, so true lands where zero was
        unrolled_mask = (np.stack([_um,_um,_um], axis=-1) == 0)
        masked_array = np.ma.masked_where(unrolled_mask, unrolled)
        return masked_array
    def masked_mean(patch, patch_mask): # ph x pw x 3, ph x pw
        if patch_mask.sum() <= 1: return None # Entirely masked (one pixel insufficient)
        masked_array = get_masked_array(patch, patch_mask)
        if verbose: print('\tNvalid pixels in patch mean:', masked_array.count())
        return np.ma.MaskedArray.mean(masked_array, axis=0)
    def masked_cov(patch, patch_mask):
        if patch_mask.sum() <= 1: return None # Entirely masked (one pixel insufficient)
        masked_array = get_masked_array(patch, patch_mask)
        return np.ma.cov(masked_array, rowvar=False)
    # Actual calculations
    means = [ masked_mean(p,m) for p,m in zip(patches,mpatches) ]
    covs = [ masked_cov(p,m) for p,m in zip(patches,mpatches) ]
    # Show means and covs if verbose
    if verbose:
        def table_print(arr):
            t = 0
            for r in range(nr):
                for c in range(nc):
                    np.set_printoptions(precision=2) # sorry
                    print(arr[t].data if not arr[t] is None else arr[t], ",", end = '')
                    t += 1
                print("\n")
        print('Means (masked)'); table_print(means)
        print('Covs (masked)');  table_print(covs)
    # Remove invalid patches
    mean_nones = [(m is None) for m in means]
    cov_nones = [(c is None) for c in covs]
    if verbose: print("IGNORING %d PATCHES" % len([m for m in means if m is None]))
    assert all(m == c for m,c in zip(mean_nones, cov_nones))
    means = [m for m in means if not m is None]
    covs  = [c for c in covs  if not c is None]
    n_valid = len(means)
    if verbose: print('N_valid', n_valid)
    # Calculate actual pairwise distances based on the moments
    channel_len = 3
    C_squared = channel_len**2
    N_valid_squared = n_valid**2
    if mode == 'central':
        D = -np.ones((n_valid,n_valid))
        # Loop pairwise over the estimated means and covariances
        for i in range(n_valid):
            for j in range(i,n_valid): # i know, +1, just checking
                # Mean distance
                mean_i = means[i]
                mean_j = means[j]
                mean_distance = np.sqrt( ((mean_i - mean_j)**2).sum() + 1e-7 ) / channel_len
                # Covariance distance
                if gamma_cov_weight < 1e-6:
                    c_dist = 0
                else:
                    cov_i = covs[i]
                    cov_j = covs[j]
                    c_dist = np.sqrt( ( np.abs(cov_i - cov_j) ).sum() + 1e-7 ) / C_squared
                # Total distance
                D[i,j] = gamma_mu_weight * mean_distance + gamma_cov_weight * c_dist
                D[j,i] = D[i,j]
    else:
        D = gaussian_prob_divergence(mode=mode, means=means, covs=covs)
    if verbose: print("\nDistances:\n", D)
    # The final complexity is the average pairwise distance between the moments of the patches
    # Note that the diagonals of D should be ~zero, and the matrix should be symmetric.
    complexity = D.mean()
    return complexity

def blockify(img, mask, block_cuts, verbose):
    """ Separates the image and its alpha mask into non-overlapping blocks.

    If the width and/or height of the image is not a multiple of 
    'block_cuts', the image is shaved from the left and/or bottom.

    Args:
        img: The input image.
        mask: The alpha mask of the image.
        block_cuts: A list [x, y] where x and y are the number of horizontal
          and vertical cuts respectively. If the height or width of 'img' is
          not a multiple of x or y, the image is shaved from the bottom and/or
          left. [3,4] by default.
        verbose: Print verbosely if True.

    Returns:
        A tuple containing:
            1) A 2D matrix of the blocks for the image.
            2) A 2D matrix of the blocks for the alpha mask.

        Each matrix has a height of 'block_cuts[0]' and width of 'block_cuts[1]'.
    """
    if verbose: print('img, mask, block_cuts shapes:', img.shape, mask.shape, block_cuts)
    N_divs_h, N_divs_w = block_cuts
    H, W, C = img.shape
    assert mask.shape[0] == img.shape[0] and img.shape[1] == img.shape[1]
    residual_h, residual_w = H % N_divs_h, W % N_divs_w
    if verbose: print('residuals', residual_h, residual_w)
    if residual_h > 0:
        img = img[:-residual_h, :, :] # Shave from bottom
        mask = mask[:-residual_h, :]
    if residual_w > 0:
        img = img[:, residual_w:, :] # Shave from left
        mask = mask[:, residual_w:]
    if verbose: print('new shapes', img.shape, mask.shape)
    Bh, Bw = H // N_divs_h, W // N_divs_w
    if verbose: print('block sizes', Bh, Bw)
    mask_blocks = skimage.util.shape.view_as_blocks(np.ascontiguousarray(mask), (Bh, Bw))
    img_blocks = channelwise_extract_blocks(img, (Bh, Bw))
    return img_blocks, mask_blocks

def channelwise_extract_blocks(I, block_shape):
    """ Separates the image into non-overlapping blocks.

    The image 'I' must be a multiple of 'block_shape'.

    Args:
        I: The input image.
        block_shape: A list [x, y] representing the shape of each block.
          The height and width of 'I' must be divisible by x and y respectively.

    Returns:
        A 2D matrix of blocks for the image.
    """
    Cs = [ skimage.util.shape.view_as_blocks(np.ascontiguousarray(I[:,:,i]), block_shape)
           for i in range(3) ]
    return np.stack(Cs, axis=-1) # NB_h x NB_w x HB x WB x C

#

def read_csv_full(path):
    """ Reads a csv file.

    The csv file must be delimited by ','.
    Note that the header will be included in the result.

    Args:
        path: The path to the csv file to read.

    Returns:
        A list of rows in the csv file where each row
        is a list of entries in the csv file.
    """
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        return [ row for row in readCSV ]

def get_row_via(targ_list, search_term, index):
    """ Gets the index where 'search_term' appears in row 'index'.

    'targ_list' represents a csv file.

    Args:
        targ_list: A 2D list representing a csv file.
          See read_csv_full().
        search_term: The value being searched for.
        index: The row of the csv file to search.
    
    Returns:
        The index (column) where 'search_term' appears or None.
    """
    for j, targ in enumerate(targ_list):
        if targ[index] == search_term:
            return j
    return None


#

class Formatter(object):
    """ An object used to prettyprint Python objects.

    Methods are defined for dict, list, and tuple which recursively
    call each other based on the objects contained in them. This
    bottoms out when an object that is not a dict, list, or tuple is
    seen; in this case, the object's __repr__ method is used.
    """
    def __init__(self):
        self.types = {}
        self.htchar = '   '
        self.lfchar = '\n'
        self.indent = 0
        self.set_formatter(object, self.__class__.format_object)
        self.set_formatter(dict, self.__class__.format_dict)
        self.set_formatter(list, self.__class__.format_list)
        self.set_formatter(tuple, self.__class__.format_tuple)

    def set_formatter(self, obj, callback):
        self.types[obj] = callback

    def __call__(self, value, **args):
        for key in args:
            setattr(self, key, args[key])
        formater = self.types[type(value) if type(value) in self.types else object]
        return formater(self, value, self.indent)

    def format_object(self, value, indent):
        return repr(value)

    def format_dict(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + repr(key) + ': ' +
            (self.types[type(value[key]) if type(value[key]) in self.types else object])(self, value[key], indent + 1)
            for key in value
        ]
        return '{%s}' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_list(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '[%s]' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_tuple(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '(%s)' % (','.join(items) + self.lfchar + self.htchar * indent)

    def print_dict(self, x):
        print(self.format_dict(x, 0))

# 

def fill_masked_pixels(I, mask, mode = 'mean', mask_thresh = 0.5, verbose = False):
    if mode == 'mean':
        #assert len(mask.shape) == 2
        #assert mask.max() <= 1.0+1e-8 and mask.min() >= -1e-8
        # Last channel is the mask -> 1 means it is a valid pixel
        bool_mask  = mask > mask_thresh
        maskf      = mask[:, :, np.newaxis]
        pixel_vals = I[bool_mask] # Unmasked pixels
        avg        = pixel_vals.mean(0)[np.newaxis, np.newaxis, :]
        replaced   = (maskf * I + (1.0 - maskf) * avg).astype(np.uint8)
        return replaced
    else:
        raise ValueError('Unknown mode ' + str(mode))



#
