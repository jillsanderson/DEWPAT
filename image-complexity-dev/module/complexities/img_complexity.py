import numpy as np, skimage, warnings
from .entropy_estimators import entropy
from scipy import fftpack as fp
from skimage.morphology import disk
from scipy.stats import entropy as scipy_discrete_shannon_entropy
from ..utils import *

#>> Measure 0: Channel-wise entropy in nats over pixels
@timing_decorator()
def discrete_pixelwise_shannon_entropy(img, alpha_mask=None, verbose=False, timing=False):
    r""" Computes the discrete Shannon entropy over individual pixel values across the image.

    This function computes the discrete Shannon entropy of each channel
    and returns the average:

    \frac{-1}{|C|} \sum_{c\in C} \sum_{s\in V_c} p(s) \log p(s)

    where
    - C is the set of channels
    - V_c is the set of pixels in channel c
    - p(s) is the frequency of pixel s

    The natural logarithm is used so the result is in units of nats.

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image.
          If alpha_mask is not None, masked pixels will be ignored.
          None by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.
    
    Returns:
        The Shannon entropy of the image. If 'timing' is True, the timing of the
        function is also returned.
    """
    if verbose: print('Computing image entropy ' + ('(alpha masked)' if not alpha_mask is None else ''))  
    if not alpha_mask is None:
        def masked_discrete_shannon(channel, first):
            channel = np.copy(channel).astype(int)
            _fake_pixel_val, alpha_mask_threshold = -1000, 0
            channel[ alpha_mask <= alpha_mask_threshold ] = _fake_pixel_val
            unique_vals, counts = np.unique(channel, return_counts=True)
            new_counts = [ c for ii, c in enumerate(counts) if not unique_vals[ii] == _fake_pixel_val ]
            if first and verbose: print('\tN_pixels before & after masking:', sum(counts), '->', sum(new_counts))
            return scipy_discrete_shannon_entropy(new_counts, base=np.e)
        shannon_entropy = np.mean([masked_discrete_shannon(img[:,:,i], i==0) for i in range(3)])
    else:
        shannon_entropy = np.mean([skimage.measure.shannon_entropy(img[:,:,i], base=np.e) for i in range(3)])
    return shannon_entropy


#>> Measure 1: Averaged channel-wise local entropy
@timing_decorator()
def channelwise_local_entropies(img, alpha_mask=None, show_locent_image=False, local_entropy_disk_size=24,
                                verbose=False, timing=False):
    r""" Computes the mean discrete Shannon entropy within local image patches.

    The shape of each patch is defined by a disk with radius 'local_entropy_disk_size'.
    The discrete Shannon entropy is computed for each patch and the average is returned:

    \frac{-1}{|C|} \sum_{c\in C} \frac{1}{|P_c|} \sum_{\zeta\in P_c} \sum_{v\in \zeta}  p(v) \log p(v)

    where
    - C is the set of channels
    - P_c is the set of patches in channel c
    - p(v) is the frequency of pixel v

    The base 2 logarithm is used so the result is in units of bits.

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image.
          If alpha_mask is not None, masked pixels will be ignored.
          None by default.
        show_locent_image: Optional; If True, display the image of 
          local entropies. False by default.
        local_entropy_disk_size: Optional; The radius of the image
          patches in pixels. 24 by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.

    Returns:
        The average local entropy. If 'timing' is True, the timing of the
        function is also returned.
    """
    if verbose: print('Computing local discrete entropies')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        current_mask_local_ent = alpha_mask
        le_img = np.array([ skimage.filters.rank.entropy(img[:,:,i], 
                                                            disk(local_entropy_disk_size),
                                                            mask=current_mask_local_ent)
                    for i in range(3) ]).mean(axis=0)
    if show_locent_image: 
        imdisplay(le_img, 'Local Entropies', colorbar=True, cmap='plasma', mask=alpha_mask)
    return np.mean(le_img)


#>> Measure 2: High frequency content via weighted average
# Note: masks, even if present, are not used here, since we would be getting irregular domain harmonics rather than Fourier coefs
# This does mean, however, that background pixels do still participate in the measure (e.g., a white dewlap on a white bg, will
# incur different frequency effects than on a black bg). 
@timing_decorator()
def mean_weighted_fourier_coef(img, 
                               alpha_mask=None, 
                               mode='mean', 
                               weight_type='l2',
                               show_fourier_image=False, 
                               verbose=False, 
                               timing=False):
    r""" Computes the frequency-weighted average of the Fourier coefficient values.

    The Fourier coefficients are weighted by the Manhattan distance from the center
    of the image, which is proportional to frequency:

    \frac{1}{Z_\gamma} \sum_{\psi_x,\,\psi_y\in \Psi} \gamma(\psi_x,\psi_y)\, \mathcal{I}_F(\psi_x,\psi_y)

    where
    - \mathcal{I}_F is the average fourier image across all channels
      (\frac{1}{|C|}\sum_{c\in C} \log \left| \mathfrak{F}[c] \right|)
    - \Psi is the set of frequency space coordinates
    - \gamma(x,y) = |x| + |y| is the Manhattan distance weights
    - Z_\gamma is the sum of the Manhattan distance weights across all
      pairs of frequency space coordinates
      (\sum_{\psi_x,\,\psi_y\in \Psi} \gamma(\psi_x,\psi_y))

    Note: masks, even if present, are not used here, since we would be getting irregular
    domain harmonics rather than Fourier coefficients. This does mean, however, that
    background pixels do still participate in the measure (e.g., a white dewlap on a
    white bg, will incur different frequency effects than on a black bg).

    Args:
        img: The input image.
        mode: Optional; the method of background handling. One of 'std' or 'mean'.
        show_fourier_image: Optional; If True, the Fourier transformed and weighted
          image will be displayed. False by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True. False by default.

    Returns:
        The weighted average of the Fourier coefficient values.
        If 'timing' is True, the timing of the function is also returned.
    """
    weight_type = weight_type.lower().strip()
    assert weight_type in ['l1', 'l2'], "Unrecognized weight type"
    assert mode in ['std', 'mean'], "Unrecognized Fourier bg handling mode"
    h, w = img.shape[0:2]
    c = h / 2 - 0.5, w / 2 - 0.5
    if verbose: print('Computing Fourier images')
    #
    if mode == 'std':
        if verbose: print('No changes to image before Fourier calculation')
    elif mode == 'mean': 
        if not alpha_mask is None: # only run if mask exists
            img = fill_masked_pixels(img, alpha_mask, mode = 'mean', verbose=verbose)
            if verbose: print('Filling masked pixels with mean unmasked value')

    fourier_images = [ fp.fft2(img[:,:,find]) for find in range(3) ]
    shifted_fourier_images = np.array([ np.fft.fftshift(fourier_image) for fourier_image in fourier_images ])
    shifted_fourier_logmag_image = np.array([ np.log( np.abs(shifted_fourier_image) )
                                        for shifted_fourier_image in shifted_fourier_images
                                    ]).transpose((1,2,0)) # Same shape as input
    avg_fourier_image = np.mean(shifted_fourier_logmag_image, axis=2)
    # Manhattan weighting from center
    if weight_type == 'l1':
        index_grid_cen = np.array([[ np.abs(i-c[0]) + np.abs(j-c[1])
                            for j in range(0,w)] for i in range(0,h)])
    elif weight_type == 'l2':
        index_grid_cen = np.array([[ np.sqrt( np.clip( (i-c[0])**2 + (j-c[1])**2, a_min=1e-8, a_max=None ) )
                            for j in range(0,w)] for i in range(0,h)])
    # Normalize weights into [0,1]. Note this doesn't matter because of the later sum normalization.
    index_grid_cen = index_grid_cen / np.max(index_grid_cen) 
    fourier_reweighted_image = (avg_fourier_image * index_grid_cen) / np.sum(index_grid_cen)
    fourier_weighted_mean_coef = np.sum( fourier_reweighted_image )
    if show_fourier_image:
        imdisplay(img, 'Altered input image (for FFT)')
        imdisplay(avg_fourier_image, 'Fourier Transform', colorbar=True, cmap='viridis')
        imdisplay(index_grid_cen, 'Fourier-space distance weights', colorbar=True, cmap='gray')
        # Avoid loss of phase information in order to view image (but note it's ignored in the metric)
        mean_shifted_fourier_images = np.mean(shifted_fourier_images, axis=0)
        reweighted_shifted_fourier_img_wp = (mean_shifted_fourier_images * index_grid_cen) / np.sum(index_grid_cen)
        real_space_reweighted_img = np.abs( fp.ifft2( np.fft.ifftshift(reweighted_shifted_fourier_img_wp)) )
        imdisplay(real_space_reweighted_img, 'Reweighted real-space image', colorbar=True, cmap='hot')
    return fourier_weighted_mean_coef


#>> Measure 3: Local (intra-)patch covariances
# Closely related to the distribution of L_2 distance between patches
# Note: vectorize over the patches? use slogdet
@timing_decorator()
def local_patch_covariance(img, alpha_mask=None, local_patch_size=20, local_covar_wstep=5, show_loccov_image=False,
                           verbose=False, timing=False):
    r""" Estimates the mean local intra-patch covariance over the image.

    \frac{1}{|P|} \sum_{p\in P} \log\left( \det\left(\widehat{C}(p)\right) + 1 \right)

    where
    - P is the set of patches
    - \widehat{C}(p) is the covariance matrix of patch p
    
    If the image is greyscale, trace is used instead of determinant.

    The patches are overlapping square windows of the image; see patches_over_channels()
    for more details. Any patch containing a masked pixel will be ignored.

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image.
          If alpha_mask is not None, any patch with a masked pixel
          will not be ignored. None by default.
        local_patch_size: Optional; The length of the square patches
          in pixels. 20 by default.
        local_covar_wstep: Optional; The step size in pixels between
          adjacent patches. 5 by default.
        show_loccov_image: Optional; If True, the local covariance
          image will be displayed. False by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.

    Returns:
        The mean local intra-patch covariance of the image. If 'timing' is True,
        the timing of the function is also returned.
    """
    if verbose: print('Computing local patch covariances')
    is_scalar = len(img) == 2
    # In the greyscale case, use trace instead of determinant
    if is_scalar: scalarize = np.trace
    else:         scalarize = np.linalg.det
    # patches \in C x H x W x Px x Py; ps = patches.shape; wt = window length squared
    patches, ps, wt = patches_over_channels(img, local_patch_size, local_covar_wstep)
    # Per patch: (1) extracts window, (2) unfolds it into pixel values, (3) computes the covariance
    if not alpha_mask is None:
        alpha_over_patches = patches_per_channel(alpha_mask, local_patch_size, local_covar_wstep)
        def masked_detcov(i,j):
            mask_patch = alpha_over_patches[i,j,:,:]
            if 0 in mask_patch: return 0 # Patches with masked pixels don't contribute
            unfolded_patch = patches[:,i,j,:,:].reshape(3, wt)
            return scalarize(np.cov( unfolded_patch ))
        if verbose: 
            print('\tPatches Size: %s (Mask Size: %s)' % 
                    (str(patches.shape),str(alpha_over_patches.shape)) )
        covariance_mat_dets = [ [ masked_detcov(i,j) for j in range(ps[2]) ] for i in range(ps[1]) ]
    else:
        if verbose: print('\tPatches Size:', patches.shape)
        #print( "Patches", [ [ patches[:,i,j,:,:].reshape(3,wt) for j in range(ps[2]) ] for i in range(ps[1]) ] )
        #print( "covs", [ [ np.cov( patches[:,i,j,:,:].reshape(3, wt) )
        #                        for j in range(ps[2]) ] for i in range(ps[1]) ] )
        #sys.exit(0)
        covariance_mat_dets = [[ scalarize(np.cov( patches[:,i,j,:,:].reshape(3, wt) ))
                                    for j in range(ps[2]) ] for i in range(ps[1]) ]
    _num_corrector = 1.0
    local_covar_img = np.log(np.array(covariance_mat_dets) + _num_corrector)
    if show_loccov_image:
        h, w, c = img.shape
        resized_local_covar_img = skimage.transform.resize(local_covar_img, output_shape=(h,w), order=3)
        imdisplay(resized_local_covar_img, 'Local Covariances', cmap = 'viridis', 
                    colorbar = True, mask = alpha_mask) 
    return np.mean(local_covar_img)


#>> Measure 4: Average gradient magnitude of the input
# Note: this computes the second-order derivatives if we're using a gradient image
# Note: even if we mask the gradient image, the gradient on the boundary will still be high
@timing_decorator()
def avg_gradient_norm(img, alpha_mask=None, use_gradient_image=False, show_gradient_img=False, verbose=False, timing=False):
    r""" Computes the mean value of the per-channel gradient magnitude over the image.

    The L2-norm of the directional derivative for each pixel is computed
    and the average is returned:

    \frac{1}{|C|\,|V_c|} \sum_{c\in C} \sum_{s\in V_c} ||\nabla I(s)||_2

    where
    - C is the set of channels
    - V_c is the set of pixels in the channel
    - I(s) is the directional derivative for pixel s

    The gradient is estimated using the Scharr transform.

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image. If alpha_mask
          is not None, any masked pixel will not contribute to the mean.
          Note that the gradient on the boundary will still be high even
          if a mask is used. None by default.
        use_gradient_image: Optional; Set to True if 'img' is a gradient image
          (the second-order derivatives will be computed in this case).
          False by default.
        show_gradient_img: Optional; If True, the gradient image will be displayed.
          False by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.
        
    Returns:
        The mean gradient magnitude. If 'timing' is True, the timing of the 
        function is also returned.
    """
    if verbose: print('Computing average edginess')
    grad_img = generate_gradient_magnitude_image(img)
    if show_gradient_img:
        gi_title = "Mean Gradient Magnitude Image" + (" (2nd order)" if use_gradient_image else "")
        imdisplay(grad_img.mean(axis=2), gi_title, cmap='plasma', colorbar=True, mask = alpha_mask)
    if not alpha_mask is None:
        grad_img[alpha_mask <= 0] = 0
    return np.mean(grad_img)


#>> Measure 5: Continuous-space Differential Shannon entropy across pixels
@timing_decorator()
def cont_pixelwise_diff_ent(img, alpha_mask=None, transform_diff_ent=False, affine_param=150, verbose=False, timing=False):
    r""" Estimates the continuous-space differential Shannon entropy across pixels.

    -\int_V p(v) \log p(v)\, dv

    where
    - V is the set of pixels
    - p is the probability density function

    This is estimated using the classic K-L k-nearest neighbor continuous
    entropy estimator. The natural logarithm is used so the result is in
    units of nats.

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image. If alpha_mask
          is not None, masked pixels will be ignored. None by default.
        transform_diff_ent: Optional; If True, the entropy value will be
          affinely transformed. False by default.
        affine_param: Optional; If 'transform_diff_ent' is True, this is
          the parameter used in the affine transform. 150 by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.

    Returns:
        The estimated pixel-wise differential Shannon entropy.
        If 'timing' is True, the timing of the function is also returned.
    """
    h, w = img.shape[0:2]
    if verbose: print('Computing continuous pixel-wise differential entropy')
    float_img = skimage.img_as_float(img).reshape(h * w, 3) # list of pixel values
    if not alpha_mask is None:
        unfolded_mask = alpha_mask.reshape(h * w)
        _str_tmp = "\tMask Sum: " + str(unfolded_mask.sum()) + ", Orig Shape: " + str(float_img.shape)
        float_img = float_img[ unfolded_mask > 0 ]
        _str_tmp += ", Masked Shape: " + str(float_img.shape)
        if verbose: print(_str_tmp)
    # Compute nearest neighbour-based density estimator
    pixelwise_diff_entropy = entropy(float_img, base=np.e)
    _oneparam_affine = lambda x, affp: (x + affp) / affp
    if transform_diff_ent: pixelwise_diff_entropy = _oneparam_affine(pixelwise_diff_entropy, affine_param)
    return pixelwise_diff_entropy


#>> Measure 6: Continuous-space Differential Shannon entropy over patches
# Note 1: this measure is not invariant to rotations of patches
# Note 2: for images with large swathes of identical patches, this method can suffer large increases
#   in computational cost (due to KD-tree construction/querying difficulties I suspect).
@timing_decorator()
def patchwise_diff_ent(img, alpha_mask=None, diff_ent_patch_size=3, diff_ent_window_step=2, max_patches_cont_DE=10000,
                       affine_param=150, transform_diff_ent=False, verbose=False, timing=False):
    r""" Estimates the differential entropy of the distribution of patches over the image.

    Similar to cont_pixelwise_diff_ent() but with patches instead of pixels:
    
    -\int_P p(\xi) \log p(\xi)\, d\xi

    where
    - P is the set of patches
    - p is the probability density function

    This is estimated using the classic K-L k-nearest neighbor continuous entropy
    estimator. The natural logarithm is used so the result is in units of nats.

    The patches are overlapping square windows of the image; see patches_over_channels()
    for more details. Any patch containing a masked pixel will be ignored.
    Each multi-channel patch is unfolded into a single vector.

    Notes:
      - this measure is not invariant to rotations of patches
      - for images with large swathes of identical patches, this method can suffer
        large increases in computational cost

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image. If alpha_mask
          is not None, patches containing a masked pixel will be ignored.
          None by default.
        diff_ent_patch_size: Optional; The length of the square patch in
          pixels. 3 by default.
        diff_ent_window_step: Optional; The step size between adjacent
          patches in pixels. 2 by default.
        max_patches_cont_DE: Optional; The maximum number of patches used
          during estimation. If the number of patches exceed this value
          (after removing masked patches), the patches will be
          randomly sampled to meet this limit. 10000 by default.
        transform_diff_ent: Optional; If True, the entropy value will be
          affinely transformed. Note that this is simply for aesthetics.
        affine_param: Optional; If 'transform_diff_ent' is True, this is
          the parameter used in the affine transform. 150 by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.

    Returns:
        The estimated pixel-wise differential Shannon entropy.
        If 'timing' is True, the timing of the function is also returned.
    """
    if verbose: print('Computing continuous patch-wise differential entropy')
    cimg = skimage.img_as_float(img) # + 1e-6 * np.random.randn(*img.shape)
    # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
    patches_dse, ps, wt = patches_over_channels(cimg, diff_ent_patch_size, diff_ent_window_step, floatify=False)
    if verbose: print('\tPatch windows size:', patches_dse.shape)
    # Gathers unfolded patches across the image
    if not alpha_mask is None:
        alpha_over_patches = patches_per_channel(alpha_mask, diff_ent_patch_size, diff_ent_window_step)
        patch_vectors = vectorize_masked_patches(patches_dse, alpha_over_patches, ps[1], ps[2])
    else:
        patch_vectors = np.array([ patches_dse[:,i,j,:,:].reshape(wt * 3) for i in range(ps[1]) for j in range(ps[2]) ])
    # Randomly resample patches to meet maximum number present
    if max_patches_cont_DE < patch_vectors.shape[0]:
        if verbose: print('\tResampling patch vectors (original size: %s)' % str(patch_vectors.shape))
        new_inds = np.random.choice(patch_vectors.shape[0], size=max_patches_cont_DE, replace=False)
        patch_vectors = patch_vectors[new_inds]
    if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
    # Compute nearest neighbour-based density estimator
    patchwise_diff_entropy = entropy(patch_vectors, base=np.e)
    _oneparam_affine=lambda x, affp: (x + affp) / affp
    if transform_diff_ent: patchwise_diff_entropy = _oneparam_affine(patchwise_diff_entropy, affine_param)
    return patchwise_diff_entropy


#>> Measure 7: Global patch-wise covariance logdet
# Note: this measure is also sensitive to patch orientation (e.g., rotating a patch will affect it)
@timing_decorator()
def patchwise_global_covar(img, alpha_mask=None, global_cov_window_size=10, global_cov_window_step=2,
                           global_cov_norm_img=False, global_cov_aff_trans=False,
                           global_cov_affine_prm=100, verbose=False, timing=False):
    r""" Computes the log-determinant of the global covariance matrix over patches in the image.

    \log\left( \det\left( \widehat{C}(P) \right) \right)

    where
    - P is the set of patches (unfolded as vectors)
    - \widehat{C}(P) is the covariance matrix of P
    
    If the image is greyscale, trace is used instead of log-determinant.

    The patches are overlapping square windows of the image; see patches_over_channels()
    for more details. Each multi-channel patch is unfolded into a single vector.
    Any patch containing a masked pixel will be ignored.

    Note: this measure is also sensitive to patch orientation (e.g., rotating
    a patch will affect it).

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image. If alpha_mask
          is not None, patches containing a masked pixel will be ignored.
          None by default.
        global_cov_window_size: Optional; The length of the square patch in
          pixels. 10 by default.
        global_cov_window_step: Optional; The step size between adjacent 
          patches in pixels. 2 by default.
        global_cov_norm_img: Optional; If True, 'img' will be normalized
          into [0,1]. False by default.
        global_cov_aff_trans: Optional; If True, the return value will be
          affinely transformed. Note that this is simply for aesthetics.
          False by default.
        global_cov_affine_prm: Optional; If 'global_cov_aff_trans' is True,
          this is the parameter used in the affine transform. 100 by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.

    Returns:
        The log-determinant/trace of the global patch-wise covariance matrix.
        If 'timing' is True, the timing of the function is also returned.
    """
    if verbose: print('Computing global patch-wise covariance')
    is_scalar = len(img.shape) == 2
    # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
    # I.e., C x N_patches_H x N_patches_V x Window_size_H x Window_size_V
    patchesgc, ps, wt = patches_over_channels(img, global_cov_window_size, global_cov_window_step, floatify=global_cov_norm_img)
    if verbose: print('\tPatch windows size:', patchesgc.shape)
    # Gathers unfolded patches across the image
    if not alpha_mask is None:
        alpha_over_patchesgc = patches_per_channel(alpha_mask, global_cov_window_size, global_cov_window_step)
        patch_vectors = vectorize_masked_patches(patchesgc, alpha_over_patchesgc, ps[1], ps[2])
    else:
        patch_vectors = np.array([ patchesgc[:,i,j,:,:].reshape(wt * 3) for i in range(ps[1]) for j in range(ps[2]) ])
    if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
    # Compute covariance matrix of the unfolded vectors
    global_cov = np.cov( patch_vectors.T )
    def _strace(c):
        t = np.trace(c)
        return np.sign(t), t
    scalarizer = (lambda c: np.linalg.slogdet(c)) if not is_scalar else (lambda c: _strace(c))
    sign, patchwise_covar_logdet = scalarizer(global_cov) 
    #patchwise_covar_logdet = np.log( np.linalg.det(global_cov) + 1.0 )
    _oneparam_affine=lambda x, affp: (x + affp) / affp
    if global_cov_aff_trans: patchwise_covar_logdet = _oneparam_affine(patchwise_covar_logdet, global_cov_affine_prm)
    return patchwise_covar_logdet


#> Measure 8: Pairwise patch EMD (i.e., mean pairwise Wasserstein distance over patches)
# TODO bugged? doesnt remove alpha masked patches
@timing_decorator()  
def pairwise_wasserstein_distance(img, alpha_mask=None, image_rescaling_factor=1.0, emd_window_size=24,
        emd_window_step=16, coordinate_aware=False, coordinate_scale=0.2, metric='sqeuclidean', 
        use_sinkhorn=True, sinkhorn_gamma=0.25, emd_visualize=True, verbose=False, timing=False):
    r""" Computes the average Wasserstein distance between image patches.

    \frac{1}{|P_C|^2} \sum_{p_c\in P_C}\sum_{q_c\in P_C} \mathcal{W}_\rho(p_c,q_c)

    where
    - P_C is the set of patches for channel C
    - \mathcal{W}_\rho is the Wasserstein distance of order \rho

    If 'coordinate_aware' is True, local normalized spatial coordinates will be
    appended to each p \in P_C such that v \in p is a vector (x, y, p1, p2, p3)
    where x and y denote the local coordinates of the pixel within the patch and
    p1, p2, p3 are the pixel values at that location. 

    If use_sinkhorn is True, the EMD will be computed using an entropy-regularized
    approximate solver (https://pythonot.github.io/all.html#ot.sinkhorn2).
    Otherwise, an exact linear program solver will be used
    (https://pythonot.github.io/all.html#ot.emd2).

    The patches are overlapping square windows of the image; see patches_over_channels()
    for more details. Any patch containing a masked pixel will be ignored.

    Args:
        img: The input image.
        alpha_mask: Optional; The alpha mask of the image. If alpha_mask
          is not None, patches containing a masked pixel will be ignored.
          None by default.
        image_rescaling_factor: Optional; The factor to resize 'img'.
          1.0 by default.
        emd_window_size: Optional; The length of the square patches in pixels.
          24 by default.
        emd_window_step: Optional; The step size in pixels between adjacent
          patches. 16 by default.
        coordinate_aware: Optional; If True, appends local normalized spatial
          coordinates to the patch elements. False by default.
        coordinate_scale: Optional; If 'coordinate_aware' is True, this
          specifies the spatial coordinate scaling. 0.2 by default.
        metric: Optional; The distance metric used. Must be "euclidean" or
          "sqeuclidean" (squared euclidean). "sqeuclidean" by default.
        use_sinkhorn: Optional; If True, an entropy-regularized approximate
          solver is used. Otherwise, an exact linear program solver is used.
          True by default.
        sinkhorn_gamma: Optional; If 'use_sinkhorn' is True, this is the weight
          on the entropy regularization term in the Sinkhorn objective. Must be
          greater than 0. 0.25 by default.
        emd_visualize: Optional; If True, displays the rescaled image and 
          up to 20 randomly selected patches. True by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.

    Returns:
        The the average Wasserstein distance. If 'timing' is True, the timing
        of the function is also returned.
    """
    import ot
    assert metric in ['euclidean', 'sqeuclidean'], "Underlying metric must be 'euclidean' or 'sqeuclidean'"
    if verbose: print('\tImage dims', img.shape)
    # Resize image
    img = skimage.transform.rescale(img, scale=image_rescaling_factor, anti_aliasing=True, multichannel=True)
    if verbose: print('\tDownscaled image dims:', img.shape)
    if emd_visualize: imdisplay(img, 'Downscaled img')
    # Extract patches
    patches_emd, ps, wt = patches_over_channels(img, emd_window_size, emd_window_step, floatify=True)
    if verbose: print('\tPatches Shape', patches_emd.shape)
    # CASE 1: using alpha mask
    if not alpha_mask is None:
        if emd_visualize: imdisplay( alpha_mask, 'Original Mask', True )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha_mask_ds = skimage.img_as_ubyte(
                                skimage.transform.rescale( 
                                    skimage.img_as_float(skimage.img_as_ubyte(alpha_mask*255)), 
                                    scale=image_rescaling_factor) )
        alpha_mask_ds[ alpha_mask_ds >  1e-8 ] = 1
        alpha_mask_ds[ alpha_mask_ds <= 1e-8 ] = 0
        if emd_visualize: imdisplay(alpha_mask_ds, 'Downscaled mask', True)
        alpha_over_patches_emd = patches_per_channel(alpha_mask_ds, emd_window_size, emd_window_step)
        patch_lists = vectorize_masked_patches(patches_emd, alpha_over_patches_emd, ps[1], ps[2], as_list=True)
    # CASE 2: no alpha channel
    else:
        patch_lists = np.array([ patches_emd[:,i,j,:,:].reshape(3, wt).T 
                                    for i in range(ps[1]) for j in range(ps[2]) ])
    if verbose: print('\tPatch lists shape:', patch_lists.shape)
    # Visualize (random or first few) patches
    if emd_visualize:  
        nr, nc, _random_vis = 5, 4, True
        n_to_vis = nr * nc
        if n_to_vis > patch_lists.shape[0]: 
            ris = np.random.choice(patch_lists.shape[0], size=n_to_vis, replace=True)
        elif _random_vis: 
            ris = np.random.choice(patch_lists.shape[0], size=n_to_vis, replace=False)
        else:             
            ris = np.array( range(n_to_vis) )
        # Index into list (N_patches x unfolded_window_size x 3)
        patches_to_vis = patch_lists[ris, :, :].reshape(n_to_vis, emd_window_size, emd_window_size, 3)
        patch_display(patches_to_vis, nr, nc, show=False, title='EMD Patches', subtitles=ris)
    # Append coordinates if desired
    if coordinate_aware:
        if verbose: print('\tUsing coordinate-aware calculations')
        n_patches = patch_lists.shape[0]
        # Refold vectorized patches back into windows
        refolded_patches = patch_lists.reshape(n_patches, emd_window_size, emd_window_size, 3)
        # Linear walk from 0 to 1
        lin_walk = np.linspace(0.0, 1.0, emd_window_size)
        # Expand linear walk to normalized patch coordinates, and rescale spatial values
        patch_of_coords = np.transpose( np.array( np.meshgrid(lin_walk, lin_walk) ), (1,2,0) ) * coordinate_scale
        # Duplicate patches for each image patch
        coord_patches = np.repeat( patch_of_coords[ np.newaxis, :, : ], n_patches, axis=0 )
        # Concatenate image patch pixel values with coordinate values
        patch_lists = np.concatenate( (coord_patches, refolded_patches), axis=3 ).reshape(n_patches, wt, 5)
        if verbose: print('\tPatch sizes with appended coords:', patch_lists.shape)
    ## Choose solver ##
    if use_sinkhorn: # Entropy-regularized approximate solver
        solver = lambda a,b,M: ot.sinkhorn2(a,b,M,sinkhorn_gamma)
    else: # Exact linear programming solver
        solver = ot.emd2
    n = patch_lists.shape[0]
    ## Function for computing EMD between two image patches ##
    def pair_emd(xs, xt, ind, should_print):
        if verbose and should_print: print('\tComputing EMD set %d/%d' % (ind+1,n))
        # Distance matrix
        M = ot.dist(xs, xt, metric=metric) #+ 1e-7 # Prevent M.max() from being zero
        max_d = M.max()
        if max_d < 1e-6: return 0.0
        M /= max_d
        # Uniform empirical delta density weightings
        a, b = ot.unif(wt), ot.unif(wt)
        # Compute EMD
        emd_final = solver(a,b,M)
        if type(emd_final) is float: return emd_final
        else: return emd_final[0]
    # Compute EMDs
    if verbose: print('\tComputing pairwise EMDs')
    emds = np.array([ pair_emd(patch_lists[i], patch_lists[j], i, j == i+1) 
                        for i in range(n) for j in range(i+1,n) ])
    if verbose: 
        _vss = ( len(emds), np.min(emds), np.max(emds), np.std(emds) )
        print('\tNum/Min/Max/Stdev EMD values: %d/%.2f/%.2f/%.2f' % _vss)
    # Combine EMDs into a single measure via averaging
    D_w = emds.mean() 
    return D_w


# Measure 9: mean-matching pairwise distances
@timing_decorator()
def patchwise_mean_dist(img, mask, pw_mnt_dist_nonOL_WS=[3,4], show_pw_mnt_ptchs=False, verbose=False, timing=False):
    r""" Computes the average pairwise distance between the first moments (mean) of patches across the image.

    \frac{1}{|C|\,|P|^2} \sum_{p_i,p_j\in P} || \widehat{\mu}(p_i) - \widehat{\mu}(p_j) ||_2

    where
    - C is the set of channels
    - P is the set of patches
    - \widehat{\mu}(p)\in\mathbb{R}^3 is the mean pixel value over patch p

    The input image is divided into non-overlapping patches. If a patch is entirely
    masked, it is removed from consideration.

    The image may be shaved from the bottom and/or left (see below).

    Args:
        img: The input image.
        mask: The alpha mask of the image.
        pw_mnt_dist_nonOL_WS: Optional; A list [x, y] where x and y are the 
          number of horizontal and vertical cuts respectively. If the height
          or width of 'img' is not a multiple of x or y, the image is shaved
          from the bottom and/or left. [3,4] by default.
        show_pw_mnt_ptchs: Optional; If True, the image and mask patches will
          be displayed. False by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.
    
    Returns:
        The average pairwise distance between the means of the patches.
        If 'timing' is True, the timing of the function is also returned.
    """
    return pairwise_moment_distances(img, mask, block_cuts=pw_mnt_dist_nonOL_WS,
            gamma_mu_weight=1.0,
            gamma_cov_weight=0.0, 
            display_intermeds=show_pw_mnt_ptchs, verbose=verbose)


# Measure 10: moment-matching pairwise distances
@timing_decorator()
def patchwise_moment_dist(img, mask, pw_mnt_dist_nonOL_WS=[3,4], show_pw_mnt_ptchs=False, gamma_mu_weight=1.0, 
                          gamma_cov_weight=1.0, verbose=False, timing=False):
    r""" Computes the average pairwise distance between the first and second moments of patches across the image.

    Similar to patchwise_mean_dist() but includes the second moment (covariance):

    \frac{1}{|P|^2} \sum_{p_i,p_j\in P}
    \frac{\gamma_\mu}{|C|} || \widehat{\mu}(p_i) - \widehat{\mu}(p_j) ||_2 +
    \frac{\gamma_C}{|C|^2} || \widehat{\Sigma}(p_i) - \widehat{\Sigma}(p_j) ||_{1,1/2}

    where
    - C is the set of channels
    - P is the set of patches
    - \gamma_\mu is the weight given to the first moment
    - \gamma_C is the weight given to the second moment
    - \widehat{\Sigma}(p)\in\mathbb{R}^{3\times 3} is the covariance matrix for patch p
    - ||M||_{1,1/2} = \sqrt{\sum_{k,\ell} |M_{k\ell}| } is the L1,1/2 matrix norm.

    The input image is divided into non-overlapping patches. If a patch is entirely
    masked, it is removed from consideration.

    The image may be shaved from the bottom and/or left (see below).

    Args:
        img: The input image.
        mask: The alpha mask of the image.
        pw_mnt_dist_nonOL_WS: Optional; A list [x, y] where x and y are the 
          number of horizontal and vertical cuts respectively. If the height
          or width of 'img' is not a multiple of x or y, the image is shaved
          from the bottom and/or left. [3,4] by default.
        show_pw_mnt_ptchs: Optional; If True, the image and mask patches will
          be displayed. False by default.
        gamma_mu_weight: Optional; The weight given to the first moment (mean).
          1.0 by default.
        gamma_cov_weight: Optional; The weight given to the second moment (covariance).
          1.0 by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True. False by default.

    Returns:
        The average pairwise distance between the first and second moments between
        the patches. If 'timing' is True, the timing of the function is also returned.
    """
    return pairwise_moment_distances(img, mask, block_cuts=pw_mnt_dist_nonOL_WS, 
            gamma_mu_weight=gamma_mu_weight,
            gamma_cov_weight=gamma_cov_weight, 
            display_intermeds=show_pw_mnt_ptchs, verbose=verbose)


# Measure 12: Pairwise distributional patch matching - Symmetric KL divergence (Jeffrey's divergence)
# Measure 13: Pairwise distributional patch matching - Wasserstein-2 metric
# Measure 14: Pairwise distributional patch matching - Hellinger distance
# Measure 15: Pairwise distributional patch matching - Bhattacharyya distance
# Measure 16: Pairwise distributional patch matching - Mean-normed FM-eigendistance (FM-ATF)
# Based on the Forstner-Moonen metric on covariance matrices and the Bhattacharyya-normalized
#   mean-matching term, similar to Abou-Moustafa and Ferrie, 2012.
@timing_decorator()
def patchwise_moment_dist_c(img, mask, mode, pw_mnt_dist_nonOL_WS=[3,4], show_pw_mnt_ptchs=False, verbose=False, timing=False):
    r""" Compute the distributional divergences between patches, based on an underlying Gaussian prior assumption.

    \frac{1}{|P|^2} \sum_{p_i,p_j\in P} D(p_i, p_j)

    where
    - P is the set of patches
    - D(p_i, p_j) is the distributional divergence between the two patches,
      assuming each patch follows a Gaussian (maximum entropy) distribution

    The following divergences are supported:
      "pw_symmetric_KL"     - pairwise Gaussian symmetrized KL divergence
      "pw_W2"               - pairwise Gaussian Wasserstein-2 divergence
      "pw_Hellinger"        - pairwise Gaussian squared Hellinger divergence
      "pw_bhattacharyya"    - pairwise Gaussian Bhattacharyya divergence
      "pw_FMATF"            - pairwise Gaussian FM-ATF
                              Based on the Forstner-Moonen metric on covariance matrices 
                              and the Bhattacharyya-normalized mean-matching term, similar 
                              to Abou-Moustafa and Ferrie, 2012.

    The input image is divided into non-overlapping patches. If a patch is entirely
    masked, it is removed from consideration.

    The image may be shaved from the bottom and/or left (see below).

    Args:
        img: The input image.
        mask: The alpha mask of the image.
        mode: The divergence used. Must be one of "pw_symmetric_KL", "pw_W2",
          "pw_Hellinger", "pw_bhattacharyya", or "pw_FMATF".
        pw_mnt_dist_nonOL_WS: Optional; A list [x, y] where x and y are the 
          number of horizontal and vertical cuts respectively. If the height
          or width of 'img' is not a multiple of x or y, the image is shaved
          from the bottom and/or left. [3,4] by default.
        show_pw_mnt_ptchs: Optional; If True, the image and mask patches will
          be displayed. False by default.
        verbose: Optional; Print verbosely if True. False by default.
        timing: Optional; Return the timing of the function if True.
          False by default.

    Returns:
        The average distributional divergence between patches.
        If 'timing' is True, the timing of the function is also returned.
    """
    return pairwise_moment_distances(img, mask, 
            block_cuts        = pw_mnt_dist_nonOL_WS, # TODO option for overlapping as well
            gamma_mu_weight   = None,
            gamma_cov_weight  = None,
            display_intermeds = show_pw_mnt_ptchs, 
            verbose           = verbose,
            mode              = mode )
