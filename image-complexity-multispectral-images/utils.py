import os, sys, numpy as np, skimage, argparse, warnings, matplotlib, csv
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D, axes3d
from skimage import io, img_as_float
from prob import gaussian_prob_divergence

import glob

### Visualization helpers ###

# Helper function for displays
def imdisplay(inim, title, colorbar=False, cmap=None, mask=None, vmin=None, vmax=None):
    """
    mask - zeros = masked (NaNs), ones = original values
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
    if colorbar: fig.colorbar(imm)

def patch_display(patches, nrows, ncols, show=False, title=None, subtitles=None, hide_axes=False):
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
    if not title is None: fig.canvas.set_window_title(title)
    if show: plt.show()

def histogram3dplot(h, e, fig=None, verbose=True):
    """
    Visualize a 3D histogram

    Adapted from:
        https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/IP/Images/ImageHistograms.html

    Args:
        h: histogram array of shape (M, N, O)
        e: list of bin edge arrays (for R, G and B)
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
    # C x H x W x Px x Py = num_channels x n_patches_vert x n_patches_horz x patch_H x patch_W
    if floatify:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = skimage.img_as_float(img)
    P = np.array([
            patches_per_channel(img[:,:,k], patch_size, window_step)
            for k in range(img.shape[2])
        ])
    if return_meta: return P, P.shape, patch_size**2
    return P

def patches_per_channel(channel, patch_size, window_step):
    # H x W x Px x Py = n_patches_vert x n_patches_horz x patch_H x patch_W
    return view_as_windows(
                np.ascontiguousarray( channel ),
                (patch_size, patch_size),
                step=window_step)

def vectorize_single_masked_patch(patches, mask, i, j, wt):
    mask_patch = mask[i,j,:,:]
    if 0 in mask_patch: return None
    unfolded_patch = patches[:,i,j,:,:].reshape(wt * patches.shape[0])
    return unfolded_patch

def vectorize_single_masked_patch_as_list(patches, mask, i, j, wt):
    mask_patch = mask[i,j,:,:]
    if 0 in mask_patch: return None
    listified_patch = patches[:,i,j,:,:].reshape(patches.shape[0], wt).T
    return listified_patch

def vectorize_masked_patches(patches, mask, H, W, as_list=False, flatten=True, remove_none=True):
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
    '''
    Generates the per-channel L2 gradient magnitude image of the input `img`.
    Each edge pixel value is divided by `divider` (enforce output in [0,1] + numerical stability).
    If `to_ubyte` is specified True, the output is converted to the ubyte numpy dtype.
    '''
    Ig_x = np.zeros(img.shape)
    Ig_y = np.zeros(img.shape)
    for k in range(img.shape[2]):
        Ig_x[:,:,k] = filters.scharr_v(img[:,:,k])
        Ig_y[:,:,k] = filters.scharr_h(img[:,:,k])
    
    Ig_x_sq, Ig_y_sq = Ig_x * Ig_x, Ig_y * Ig_y
    gradient_img = np.sqrt(Ig_x_sq + Ig_y_sq) / divider # to remain in [0,1]
    # print('GradImg min/max: %.2f/%.2f' % (gradient_img.min(), gradient_img.max()))
    if to_ubyte:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gradient_img = skimage.img_as_ubyte(gradient_img)
    return gradient_img

def gaussian_blur(image, sigma):
    '''
    Performs Gaussian blurring on the input with standard deviation sigma.
    '''
    return np.clip(filters.gaussian(image, sigma=sigma, multichannel=True), 0.0, 1.0)

def to_perceptual_greyscale(img):
    from skimage.color import rgb2gray
    new_img = dupe_channel( rgb2gray(img) )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_img = skimage.img_as_ubyte( new_img )
    return new_img

def to_avg_greyscale(img):
    new_img = dupe_channel( np.clip(np.mean(img / 255, axis=-1), 0, 1) )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_img = skimage.img_as_ubyte( new_img )
    return new_img

def conv_to_ubyte(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_ubyte( img )
    return img

### Timing helpers ###

# Note: calling this as a decorator actually runs it, so that
# replacer is the actual decorator used.
def timing_decorator(do_timing):
    def decorator(wrapped_func):
        if do_timing:
            def replacement(*args, **kwargs):
                start = timer()
                output = wrapped_func(*args, **kwargs)
                end = timer()
                return output, (end - start)
            return replacement
        else:
            return wrapped_func
    return decorator

### Array Helpers ###

def dupe_channel(c):
    return combine_channels([c, c, c])

def combine_channels(channels):
    return np.stack( channels, axis = -1 )

### Colour Helpers ###

def color_multiinterpolator(color_list, times=None):
    '''
    Produces a color function f : [0,1] -> R^3 that linearly interpolates over color_list.
    Color_list contains a list of color arrays, with length n.
    times, if given, must be of length n-2 and each entry has 0 < t_i < 1
    '''
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


def plotDimensionallyReducedVectorsIn2D(vectors, method='pca', point_labels=None, verbose=True,
                                        manifold_learning_options={}, colors=None):
    '''
    Given a set of n-dimensional vectors, dimensionally reduce the set to 2D and plot it.
    i.e. V is an m x n matrix.
    We can color based on the point labels if given. Must be a list of
    '''
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
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
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

def load_helper(image_path, verbose=True, blur_sigma=None, apply_alpha_to_rgb_channels=True):
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

def load_color_image(im_path):
    
    nChannels = 0
    image = img_as_float(io.imread(im_path))
    
    nChannels = image.shape[2]
    
    if (nChannels == 4):
        im_mask = image[:,:,-1]
        im_mask[ im_mask <= 0 ] = 0
        im_mask[ im_mask  > 0 ] = 1
        image = image[:,:,0:3]
    else:
        im_mask = None   

    return image, im_mask

def load_mspec_image(im_path, verbose):
            
    if (type(im_path) is list):
        image,im_mask = load_im_stack(im_path)
        
        n_channels = image.shape[2]
        
        if (n_channels == 0):
            raise NameError("No images were found in file: " + im_path)
        if (verbose):
            if (im_mask is None):
                print (str(n_channels) + " images were found. No mask.")
            else:
                print (str(n_channels) + " images were found + mask.")
    else:    
        file_ext = im_path[-4:]

        if (file_ext == ".txt"):
            mask_found = 0
            n_channels = 0
            im_mask = None

            with open(im_path) as f:
                im_filenames = f.read().split('\n')
            
            image,im_mask = load_im_stack(im_filenames)
        
            n_channels = image.shape[2]
            
            if (n_channels == 0):
                raise NameError("No images were found in file: " + im_path)
            if (verbose):
                if (im_mask is None):
                    print (str(n_channels) + " images were found. No mask.")
                else:
                    print (str(n_channels) + " images were found + mask.")

        elif (file_ext == ".tif"):

            image = io.imread(im_path)
            
            if (image.shape[0] < image.shape[1]) and (image.shape[0] < image.shape[2]):
                image = np.transpose(image,(1,2,0))

            im_mask = np.ones(image.shape[0:2])

            if (np.any(np.isnan(image[:,:,0]))):
                im_mask[np.isnan(image[:,:,0])] = 0
    
    return image,im_mask


def load_im_stack(im_filenames):
    
    mask_found = 0
    n_channels = 0
    
    if (len(im_filenames) < 2):
        raise TypeError('A multispectral image must have at least two layers.')
    
    for filename in im_filenames:
        if not (filename[-4:] == ".png"):
            raise TypeError('Image format is not supported for multispectral image processing.')
        temp_im = io.imread(filename)

        if (n_channels == 0):
            image = temp_im[:,:,0]
        else:
            image = np.dstack((image, temp_im[:,:,0]))

        n_channels += 1

        if ((mask_found == False) and (temp_im.shape[2] > 3)):
            im_mask = temp_im[:,:,-1]
            im_mask = np.copy(im_mask).astype(int)
            im_mask[im_mask <= 0] = 0
            im_mask[im_mask > 0] = 1
            mask_found = True

    return image, im_mask


def load_csv_data(csv_path):
    
    file_ext = csv_path[-4:]
    
    if (file_ext == ".csv"):
        csvLines = read_csv_full(csv_path)
        del csvLines[0]
               
        im_filename_stacks = []
        currStackIndex = -1
        for line in csvLines:
            
            if not (line[2][-4:] == ".png"):
                raise TypeError('Image format is not supported for multispectral image processing.')
                
            if ((int)(line[1]) > currStackIndex):
                im_filename_stacks.append([line[2]])
                currStackIndex +=1
            else:
                im_filename_stacks[currStackIndex].append(line[2])

    print(len(im_filename_stacks),"image stacks have been found.")
    
    return im_filename_stacks


def convert_im_stack(im_path):
    # Accepts directory + image  name without suffix and returns a ndarray where each slice of depth corresponds to 1 multispectral image.
    # Returns a mask if any of the relevant images in directory have alpha channel (e.g. png images).
    
    nChannels = 0
    mask_found = 0
    im_mask = None
    
    for im_file in glob.glob(im_path+"*"):
        image = io.imread(im_file)
        if (nChannels == 0):
            im_stack = image[:,:,0]
        else:
            im_stack = np.dstack((im_stack,image[:,:,0]))
            
        nChannels += 1
        
        if ((mask_found == 0) and (image.shape[2] > 3)):
            im_mask = image[:,:,-1]
            im_mask = np.copy(im_mask).astype(int)
            im_mask[ im_mask <= 0 ] = 0
            im_mask[ im_mask  > 0 ] = 1
            mask_found = 1

    if (nChannels == 0): 
        raise NameError("No images were found in path: " + im_path)
    
    if (im_mask is None):
        print (str(nChannels) + " images were found. No mask.")
    else:
        print (str(nChannels) + " images were found + mask.")

    return im_stack, im_mask


### Block extraction and pairwise moment comparison methods ###

def pairwise_moment_distances(img, mask, block_cuts, gamma_mu_weight, gamma_cov_weight, display_intermeds, verbose, mode='central'):
    """
    Divides the input img into non-overlapping blocks.
    Note: if a patch is *entirely* masked, it is removed from consideration.

    TODO: options for other matrix norms in central moment distance.
    """
    #assert img.max() > 1, "Unexpected pixel scaling encountered"
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
        unrolled = patch.reshape(-1, img.shape[2])
        _um = patch_mask.reshape(unrolled.shape[0])
        # In numpy, we mask/ignore values where the mask value is True (rather than False)
        # So notice we *invert* the mask, so true lands where zero was
        unrolled_mask = (np.tile(_um,[img.shape[2],1]).T == 0)
        masked_array = np.ma.masked_where(unrolled_mask, unrolled)
        return masked_array
    def masked_mean(patch, patch_mask): # ph x pw x n_channels, ph x pw
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
    channel_len = img.shape[2]
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
    """
    Note: this method shaves off some rows/columns to get the block size to fit.
    """
    if verbose: print('img, mask, block_cuts shapes:', img.shape, mask.shape, block_cuts)
    N_divs_h, N_divs_w = block_cuts
    H, W, C = img.shape
    assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]
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
    Cs = [ skimage.util.shape.view_as_blocks(np.ascontiguousarray(I[:,:,i]), block_shape)
           for i in range(I.shape[2]) ]
    return np.stack(Cs, axis=-1) # NB_h x NB_w x HB x WB x C

#

def read_csv_full(path):
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        return [ row for row in readCSV ]

def get_row_via(targ_list, search_term, index):
    for j, targ in enumerate(targ_list):
        if targ[index] == search_term:
            return j
    return None


#

class Formatter(object):
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
