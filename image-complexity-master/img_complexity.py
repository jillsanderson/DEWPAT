import os, sys, numpy as np, skimage, argparse, warnings
import matplotlib.pyplot as plt, entropy_estimators as EE_cont
from scipy import fftpack as fp
from skimage.morphology import disk
from skimage.util import view_as_windows
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from scipy.stats import entropy as scipy_discrete_shannon_entropy 
from utils import *

'''
Usage: python img_complexity.py <input>
    where input is either a folder of images or a single image.
By default, computes all complexity measures (except EMD); if any specific ones are given, only those are used.
Note: operates in RGB color space.
'''

overall_desc = 'Computes some simple measures of image complexity.'
parser = argparse.ArgumentParser(description=overall_desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Input image
parser.add_argument('input', type=str, help='Input: either a folder or an image')
# Meta-arguments
parser.add_argument('--verbose', dest='verbose', action='store_true',
        help='Whether to print verbosely while running')
parser.add_argument('--ignore_alpha', dest='ignore_alpha', action='store_true',
        help='Whether to ignore the alpha mask channel')
parser.add_argument('--timing', dest='timing', action='store_true',
        help='Whether to measure and print timing on each function')
parser.add_argument('--mspec', dest='is_mspec', action='store_true',
    help='Whether to treat the input as a multispectral image')
# Preprocessing (resize, blur, greyscale, etc...)
parser.add_argument('--blur', type=float, default=0.0,
    help='Specify Gaussian blur standard deviation applied to the image')
parser.add_argument('--greyscale', type=str, default="none",
    help='Specify greyscale conversion: one of "human", "avg", or "none".')
parser.add_argument('--resize', type=float, default=1.0,
    help='Specify scalar resizing value. E.g., 0.5 halves the image size; 2 doubles it.')
# Specifying complexity measures to use
group_c = parser.add_argument_group('Complexities Arguments',
        description=('Controls which complexity measures to utilize. ' + 
                     'By default, computes all complexity measures (except EMD); ' + 
                     'if any specific ones are given, only those are used.') )
group_c.add_argument('--discrete_global_shannon',   
        dest='discrete_global_shannon', action='store_const', 
        const=0, default=None,
        help='Whether to use the discrete pixel-wise global Shannon entropy measure')
group_c.add_argument('--discrete_local_shannon',    
        dest='discrete_local_shannon', action='store_const', 
        const=1, default=None,
        help='Whether to use the discrete Shannon entropy across image patches measure')
group_c.add_argument('--weighted_fourier',          
        dest='weighted_fourier', action='store_const', 
        const=2, default=None,
        help='Whether to use the frequency-weighted mean Fourier coefficient measure')
group_c.add_argument('--local_covars',              
        dest='local_covars', action='store_const', 
        const=3, default=None,
        help='Whether to use the average log-determinant of covariances across image patches')
group_c.add_argument('--grad_mag',                  
        dest='grad_mag', action='store_const', 
        const=4, default=None,
        help='Whether to use the mean gradient magnitude over the image')
group_c.add_argument('--diff_shannon_entropy',                  
        dest='diff_shannon_entropy', action='store_const', 
        const=5, default=None,
        help='Whether to use the differential Shannon entropy computed over continuous-valued image pixels')
group_c.add_argument('--diff_shannon_entropy_patches',                  
        dest='diff_shannon_entropy_patches', action='store_const', 
        const=6, default=None,
        help='Whether to use the differential Shannon entropy computed over continuous-valued image patches')
group_c.add_argument('--global_patch_covar',                  
        dest='global_patch_covar', action='store_const', 
        const=7, default=None,
        help='Whether to use the global patch-wise log-determinant of the covariance of unfolded patches across the image')
group_c.add_argument('--pairwise_emd',                  
        dest='pairwise_emd', action='store_const', 
        const=8, default=None,
        help='Whether to use the pairwise Wasserstein distance across image patches')
group_c.add_argument('--pairwise_mean_distances',                  
        dest='pairwise_mean_distances', action='store_const', 
        const=9, default=None,
        help='Whether to use the pairwise distance between mean pixel values across image patches')
group_c.add_argument('--pairwise_moment_distances',                  
        dest='pairwise_moment_distances', action='store_const', 
        const=10, default=None,
        help='Whether to use the pairwise distances between first two moments (mean and covariance) across image patches')
group_c.add_argument('--wavelet_details',
        dest='wavelet_details', action='store_const',
        const=11, default=None,
        help='Whether to use the absolute values of the wavelet detail coefficients to measure complexity')
group_c.add_argument('--pwg_jeffreys_div',
        action='store_const',
        const=12, default=None,
        help='Whether to use the pairwise Gaussian symmetrized KL divergence across image patches')
group_c.add_argument('--pwg_w2_div',
        action='store_const',
        const=13, default=None,
        help='Whether to use the pairwise Gaussian Wasserstein-2 divergence across image patches')
group_c.add_argument('--pwg_hellinger_div',
        action='store_const',
        const=14, default=None,
        help='Whether to use the pairwise Gaussian squared Hellinger divergence across image patches')
group_c.add_argument('--pwg_bhattacharyya_div',
        action='store_const',
        const=15, default=None,
        help='Whether to use the pairwise Gaussian Bhattacharyya divergence across image patches')
group_c.add_argument('--pwg_fmatf_div',
        action='store_const',
        const=16, default=None,
        help='Whether to use the pairwise Gaussian FM-ATF divergence across image patches')

# Algorithm parameters
group_p = parser.add_argument_group('Algorithmic parameters',
            description='Options controlling parameters of complexity estimation algorithms')
group_p.add_argument('--local_cov_patch_size', type=int, default=20,
    help='Patch size for local covariance calculations')
group_p.add_argument('--local_covar_wstep', type=int, default=5,
    help='The step-size (stride) for the local covariance calculation')
group_p.add_argument('--sinkhorn_emd', action='store_true', 
    help='Specify to compute the entropy-regularized Sinkhorn approximation, rather than the exact EMD via linear programming')
group_p.add_argument('--emd_ignore_coords', action='store_true',
    help='Specify to avoid appending local normalized spatial coordinates to patch elements')
group_p.add_argument('--squared_euc_metric', action='store_true',
    help='Specify to use squared Euclidean rather than Euclidean underlying metric for the EMD')
group_p.add_argument('--emd_downscaling', type=float, default=0.2,
    help='Specify image downscaling factor for EMD calculations')
group_p.add_argument('--sinkhorn_regularizer', type=float, default=0.25,
    help='Specify Sinkhorn entropy regularization weight coefficient')
group_p.add_argument('--emd_coord_scaling', type=float, default=0.2,
    help='Specify spatial coordinate scaling for EMD calculations, which controls the relative balance between pixel vs image space distance')
group_p.add_argument('--wt_threshold_percentile', type=float, default=99,
    help='Controls the threshold percentile for the wavelet transform-based method')
group_p.add_argument('--wt_n_levels', type=int, default=4,
    help='Controls the number of decomposition levels used by the discrete wavelet transform')
group_p.add_argument('--wt_mother_wavelet', type=str, default='haar',
    help='Controls the type of mother wavelet used by the discrete wavelet transform')
group_p.add_argument('--gamma_mu_weight', type=float, default=1.0,
    help='Specifies the weight on the mu distance in the patch moment measure')
group_p.add_argument('--gamma_cov_weight', type=float, default=1.0,
    help='Specifies the weight on the covariance matrix distance in the patch moment measure')
group_p.add_argument('--pw_mnt_dist_nonOL_WS', type=str, default="3,4", 
    help='Specifies the number of patches when discretizing for the pairwise patch distance measures')

# Display options
group_v = parser.add_argument_group('Visualization Arguments',
            description='Options for viewing intermediate computations.')
group_v.add_argument('--show_fourier',      
        dest='show_fourier', action='store_true',
        help='Whether to display the Fourier transformed and weighted image')
group_v.add_argument('--show_local_ents',   
        dest='show_locents', action='store_true',
        help='Whether to display the image of local entropies')
group_v.add_argument('--show_local_covars', 
        dest='show_local_covars', action='store_true',
        help='Whether to display an image of the local covariances')
group_v.add_argument('--show_pw_mnt_ptchs', 
        dest='show_pw_mnt_ptchs', action='store_true',
        help='Whether to display an intermediate blocks of the pairwise moment comparator methods')
group_v.add_argument('--show_gradient_img', 
        dest='show_gradient_img', action='store_true',
        help='Whether to display the gradient magnitude image')
group_v.add_argument('--show_emd_intermeds', 
        dest='emd_visualize', action='store_true',
        help='Whether to visualize intermediate computations in the EMD method')
group_v.add_argument('--show_img',          
        dest='show_img', action='store_true',
        help='Whether to display the input image')
group_v.add_argument('--show_dwt',
        dest='show_dwt', action='store_true',
        help='Whether to display the discrete wavelet transform intermediaries')
group_v.add_argument('--show_all',          
        dest='show_all', action='store_true',
        help='Whether to display all of the above images')
# Gradient image usage
group_g = parser.add_argument_group('Gradient Image Input Arguments')
group_g.add_argument('--use_grad_only',     
        dest='use_grad', action='store_true',
        help='Whether to use the gradient of the image instead of the image itself')
group_g.add_argument('--use_grad_too',      
        dest='use_grad_too', action='store_true',
        help='If specified, computes complexities of both the original and the gradient image')
#> Final parsing <#
args = parser.parse_args()

# Names of complexity measures
S_all = [ 'Pixelwise Shannon Entropy', 'Average Local Entropies',
          'Frequency-weighted Mean Coefficient', 'Local Patch Covariance',
          'Average Gradient Magnitude', 'Pixelwise Differential Entropy',
          'Patchwise Differential Entropy', 'Global Patch Covariance',
          'Pairwise Wasserstein Distance', 'Pairwise Mean Distances',
          'Pairwise Moment Distances', 'Wavelet Detail Coef',
          'PWG Jeffreys Divergence', 'PWG Wass2 Divergence', 'PWG Hellinger Divergence',
          'PWG Bhattacharyya Divergence', 'PWG FM-ATF Divergence' ]
# Discern which measures to use
input_vals = [ args.discrete_global_shannon, args.discrete_local_shannon, 
               args.weighted_fourier, args.local_covars, 
               args.grad_mag, args.diff_shannon_entropy, 
               args.diff_shannon_entropy_patches, args.global_patch_covar, 
               args.pairwise_emd, args.pairwise_mean_distances,
               args.pairwise_moment_distances, args.wavelet_details,
               args.pwg_jeffreys_div, args.pwg_w2_div, args.pwg_hellinger_div,
               args.pwg_bhattacharyya_div, args.pwg_fmatf_div]
assert len(S_all) == len(input_vals)
if all(map(lambda k: k is None, input_vals)):
    # No metric specified
    if args.verbose: print('Using all complexity measures except Pairwise EMD')
    EMD_index = 8
    S = S_all.copy() # force S_all to still hold all names (not modified via reference)
    input_vals = list(range(len(input_vals)))
    input_vals.remove( EMD_index )
    S.remove( S[EMD_index] ) # remove EMD from S (list of metrics to use as strings + img path)
else:
    # Else, gather the specified metrics
    S = [ S_all[i] for i in range(len(input_vals)) if i in input_vals ]
    if args.verbose: print('Using:', ", ".join(S))
# S_all = string forms of each complexity
# S = [img+path] + string forms of complexities to use
# complexities_to_use = list of ints denoting which complexities to use
S = ['Image path'] + S
complexities_to_use = [ val for val in input_vals if not val is None ]

#------------------------------------------------------------------------------#

def compute_complexities(impath,    # Path to input image file
        complexities_to_use,        # List of ints describing which complexity metric to apply
        print_mode=None,            # Switch between multi-image CSV mode and single-image printing mode
        ### Measure parameters ###
        # Local entropy options
        local_entropy_disk_size=24, # Patch size for local entropy calculations
        # Local covariance options
        #local_patch_size=20,        # Patch size for local covariance calculations
        #local_covar_wstep=5,        # The step-size (stride) for the local covariance calculation
        # Shared Patch and pixelwise differential entropy options
        transform_diff_ent=False,   # Whether to affinely transform the differential entropy
        affine_param=150,           # Parameter used in affine transform for differential entropy
        # Patchwise differential entropy options
        diff_ent_patch_size=3,      # Size of patch to unfold for differential entropy calculations
        diff_ent_window_step=2,     # Patch extraction step size for differential entropy estimation
        max_patches_cont_DE=10000,  # Max num patches for continuous diff ent estimation (resampled if violated)
        # Global covariance options
        global_cov_window_size=10,  # Size of patch for global patch covariances 
        global_cov_window_step=2,   # Patch extraction step size for global patch covariance computations
        global_cov_norm_img=False,  # Whether to normalize the image into [0,1] before computing global covariances
        global_cov_aff_trans=False, # Whether to affinely transform the output logdet covariances
        global_cov_affine_prm=100,  # Parameter used in affine transform for global patch-wise logdet covariances
        # Pairwise EMD parameters
        emd_window_size=24,         # Window size for pairwise EMD
        emd_window_step=16,         # Window step for pairwise EMD
        # Mean and moment distances parameters
        #pw_mnt_dist_nonOL_WS=(3,4), # Non-overlapping patches to segment the image into for moment distances
        ### Visualization Options ###
        use_gradient_image = False, # Whether to use channel-wise gradient image instead of the raw input
        show_fourier_image = False, # Whether to display Fourier-based images
        show_locent_image = False,  # Whether to display the local entropies
        show_loccov_image = False,  # Whether to display the local covariances
        show_gradient_img = False,  # Shows 2nd derivatives if use_gradient_image is true
        show_pw_mnt_ptchs = False,  # Show the patches used to compute the moments (for measures 9 and 10)
        show_dwt = False,           # Show the DWT intermediaries
        display_image = False,      # Whether to display the image under consideration
        verbose=False,               # Whether to print verbosely when running
        is_mspec=False             # Whether to treat the input as a multispectral image
    ):
    '''
    Computes pixelwise entropy, average filtered entropy, weighted frequency content,
        mean local covariance logdets, and average gradient magnitude (edginess).

    If use_gradient_image is specified, the entire function is run on the gradient
        magnitude image of the original image, as if it were the input. The average
        gradient measure in this case, is actually the second-order gradient
        of the original image. This applies when show_gradient_img is True as well.

    The affine parameters for the patch-wise differential entropy and covariance control an order-preserving (monotonic)
        transform for the final output. This is simply for aesthetics, to make the values positive and of smaller magnitude.
        This can be turned off via `transform_diff_ent` and `global_cov_aff_trans`.

    Notes:
        RGB color space is used, which may not be perceptually ideal.
        These measures may be image resolution or size dependent.
        May want to consider measures based on the structure tensor.
        May want to consider whether to log the covariance determinant or not.
    '''
    assert print_mode in [None, 'single', 'compact'], 'print_mode must be None, "single", or  "compact"'
    # Helper for single parameter affine transform
    _oneparam_affine = lambda x, affp: (x + affp) / affp
    # Read original image
    if verbose: print('Reading image:', impath)
    # Check if color image or mspec image
    is_color = not args.is_mspec
    if is_color:
        img,img_mask = load_color_image(impath)
    else:
        img,img_mask = load_mspec_image(impath, verbose)
        # UGLY FIX: NEED TO CLEAN UP
        if (type(impath) is list):
            impath = impath[0]
    
    n_channels = img.shape[2]

    # Downscale image, if desired
    resize_factor_main = args.resize
    assert resize_factor_main > 0.0, "--resize must be positive"
    running_resize = False
    if abs(resize_factor_main - 1.0) > 1e-4:
        running_resize = True
        if verbose: print("Orig shape", img.shape, "| Resizing by", resize_factor_main)
        img = skimage.transform.rescale(img, scale=resize_factor_main, 
                anti_aliasing=True, multichannel=True)
        img = conv_to_ubyte(img)
        if verbose: print("Resized dims:", img.shape)
        if (img_mask is not None):
            img_mask = skimage.transform.rescale(img_mask, scale=resize_factor_main, 
                anti_aliasing=True, multichannel=False)
            img_mask[ img_mask > 0] = 1
            img_mask[ img_mask <= 0] = 0
            if (False and is_color):
                imdisplay(img_mask, 'alpha_layer', colorbar=True, cmap='plasma')
                imdisplay(img, 'layers', colorbar=True, cmap='plasma')
                plt.show()
                   
    # Handle alpha transparency
    
    using_alpha_mask = not (img_mask is None)
    if using_alpha_mask:
        if (False and is_color): # Display mask
            imdisplay(img_mask, 'image_mask', colorbar=True, cmap='plasma')
            plt.show()

    # Ignore the alpha channel, if desired
    if args.ignore_alpha:
        using_alpha_mask = False
        img_mask = None
    
    # Convert image to greyscale, if desired
    is_scalar = False
    if (is_color):
        gs_type = args.greyscale.lower()
        assert gs_type in ["none", "human", "avg"], 'Use one of --greyscale none/avg/human'
        if gs_type == 'human':
            if verbose: print("Greyscaling image (perceptual)")
            img = to_perceptual_greyscale(img)
            is_scalar = True
        elif gs_type == 'avg':
            if verbose: print("Greyscaling image (channel mean)")
            img = to_avg_greyscale(img)
            is_scalar = True
        
    # Blur image, if desired
    blur_sigma = args.blur
    assert blur_sigma >= 0.0, "Untenable blur kernel width"
    if blur_sigma > 1e-5:
        if verbose: print("\tBlurring with sigma =", blur_sigma)
        #bb = gaussian_blur(img, blur_sigma)
        #print(bb.max(), bb.min())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = skimage.img_as_ubyte( gaussian_blur(img, blur_sigma) )
    
    # Switch to channel-wise gradient magnitude image if desired
    if use_gradient_image:
        if verbose: print('Using gradient magnitude image')
        img = generate_gradient_magnitude_image(img, to_ubyte=True) # Overwrite
    
    # Perform masking directly on the image (to destroy details hidden behind the alpha channel)
    if using_alpha_mask and not args.ignore_alpha:
        img[ img_mask <= 0 ] = 0
    if (display_image and is_color): imdisplay(img, 'Image')
    if verbose:
        print('Image Shape:', img.shape)
        print('Channelwise Min/Max')
        for i in range(n_channels):
            print(i, 'Min:', np.min(img[:,:,i]),'| Max:',np.max(img[:,:,i]))  
    
    # Image dimensions and center
    h, w = img.shape[0:2]
    c = h / 2 - 0.5, w / 2 - 0.5
    # Tracking code
    complexities, computed_names, timings = [], [], []
    def add_new(val, ind):
        if args.timing:
            val, time = val 
            timings.append(time)
        complexities.append(val)
        computed_names.append(S_all[ind])

    #####################################
    ### Computing complexity measures ###
    #####################################

    #>> Measure 0: Channel-wise entropy in nats over pixels
    if 0 in complexities_to_use:
        @timing_decorator(args.timing)
        def discrete_pixelwise_shannon_entropy(img):
            if verbose: print('Computing image entropy ' + ('(alpha masked)' if using_alpha_mask else ''))  
            if using_alpha_mask:
                def masked_discrete_shannon(channel, first):
                    channel = skimage.img_as_int(np.copy(channel))
                    _fake_pixel_val, alpha_mask_threshold = -1000, 0
                    channel[ img_mask <= alpha_mask_threshold ] = _fake_pixel_val
                    unique_vals, counts = np.unique(channel, return_counts=True)
                    new_counts = [ c for ii, c in enumerate(counts) if not unique_vals[ii] == _fake_pixel_val ]
                    if first and verbose: print('\tN_pixels before & after masking:', sum(counts), '->', sum(new_counts))
                    return scipy_discrete_shannon_entropy(new_counts, base=np.e)
                shannon_entropy = np.mean([masked_discrete_shannon(img[:,:,i], i==0) for i in range(n_channels)])
            else:
                shannon_entropy = np.mean([skimage.measure.shannon_entropy(img[:,:,i], base=np.e) for i in range(n_channels)])
            return shannon_entropy
        add_new(discrete_pixelwise_shannon_entropy(img), 0)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #>> Measure 1: Averaged channel-wise local entropy
    if 1 in complexities_to_use:
        @timing_decorator(args.timing)
        def channelwise_local_entropies(img):
            if verbose: print('Computing local discrete entropies')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_mask_local_ent = img_mask if using_alpha_mask else None
                le_img = np.array([ skimage.filters.rank.entropy(img[:,:,i], 
                                                                 disk(local_entropy_disk_size),
                                                                 mask=current_mask_local_ent)
                            for i in range(n_channels) ]).mean(axis=0)
            if (show_locent_image and is_color): 
                imdisplay(le_img, 'Local Entropies', colorbar=True, cmap='plasma', mask=img_mask)
            return np.mean(le_img)
        add_new(channelwise_local_entropies(img), 1)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #>> Measure 2: High frequency content via weighted average
    # Note: masks, even if present, are not used here, since we would be getting irregular domain harmonics rather than Fourier coefs
    # This does mean, however, that background pixels do still participate in the measure (e.g., a white dewlap on a white bg, will
    # incur different frequency effects than on a black bg). 
    if 2 in complexities_to_use:
        @timing_decorator(args.timing)
        def mean_weighted_fourier_coef(img):
            if verbose: print('Computing Fourier images')
            fourier_images = [ fp.fft2(img[:,:,find]) for find in range(n_channels) ]
            shifted_fourier_images = np.array([ np.fft.fftshift(fourier_image) for fourier_image in fourier_images ])
            shifted_fourier_logmag_image = np.array([ np.log( np.abs(shifted_fourier_image) )
                                                for shifted_fourier_image in shifted_fourier_images
                                            ]).transpose((1,2,0)) # Same shape as input
            avg_fourier_image = np.mean(shifted_fourier_logmag_image, axis=2)
            # Manhattan weighting from center
            index_grid_cen = np.array([[ np.abs(i-c[0]) + np.abs(j-c[1])
                                for j in range(0,w)] for i in range(0,h)])
            # Normalize weights into [0,1]. Note this doesn't matter because of the later sum normalization.
            index_grid_cen = index_grid_cen / np.max(index_grid_cen) 
            fourier_reweighted_image = (avg_fourier_image * index_grid_cen) / np.sum(index_grid_cen)
            fourier_weighted_mean_coef = np.sum( fourier_reweighted_image )
            if (show_fourier_image and is_color):
                imdisplay(avg_fourier_image, 'Fourier Transform', colorbar=True, cmap='viridis')
                imdisplay(index_grid_cen, 'Fourier-space distance weights', colorbar=True, cmap='gray')
                # Avoid loss of phase information in order to view image (but note it's ignored in the metric)
                mean_shifted_fourier_images = np.mean(shifted_fourier_images, axis=0)
                reweighted_shifted_fourier_img_wp = (mean_shifted_fourier_images * index_grid_cen) / np.sum(index_grid_cen)
                real_space_reweighted_img = np.abs( fp.ifft2( np.fft.ifftshift(reweighted_shifted_fourier_img_wp)) )
                imdisplay(real_space_reweighted_img, 'Reweighted real-space image', colorbar=True, cmap='hot')
            return fourier_weighted_mean_coef
        add_new(mean_weighted_fourier_coef(img), 2)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #>> Measure 3: Local (intra-)patch covariances
    # Closely related to the distribution of L_2 distance between patches
    # Note: vectorize over the patches? use slogdet
    if 3 in complexities_to_use:
        local_patch_size = args.local_cov_patch_size
        local_covar_wstep = args.local_covar_wstep
        @timing_decorator(args.timing)
        def local_patch_covariance(img):
            if verbose: print('Computing local patch covariances')
            # In the greyscale case, use trace instead of determinant
            if is_scalar: scalarize = np.trace
            else:         scalarize = np.linalg.det
            # patches \in C x H x W x Px x Py; ps = patches.shape; wt = window length squared
            patches, ps, wt = patches_over_channels(img, local_patch_size, local_covar_wstep)
            # Per patch: (1) extracts window, (2) unfolds it into pixel values, (3) computes the covariance
            if using_alpha_mask:
                alpha_over_patches = patches_per_channel(img_mask, local_patch_size, local_covar_wstep)
                def masked_detcov(i,j):
                    mask_patch = alpha_over_patches[i,j,:,:]
                    if 0 in mask_patch: return 0 # Patches with masked pixels don't contribute
                    unfolded_patch = patches[:,i,j,:,:].reshape(n_channels, wt)
                    return scalarize(np.cov( unfolded_patch ))
                if verbose: 
                    print('\tPatches Size: %s (Mask Size: %s)' % 
                            (str(patches.shape),str(alpha_over_patches.shape)) )
                covariance_mat_dets = [ [ masked_detcov(i,j) for j in range(ps[2]) ] for i in range(ps[1]) ]
            else:
                if verbose: print('\tPatches Size:', patches.shape)
                #print( "Patches", [ [ patches[:,i,j,:,:].reshape(n_channels,wt) for j in range(ps[2]) ] for i in range(ps[1]) ] )
                #print( "covs", [ [ np.cov( patches[:,i,j,:,:].reshape(n_channels, wt) )
                #                        for j in range(ps[2]) ] for i in range(ps[1]) ] )
                #sys.exit(0)
                covariance_mat_dets = [[ scalarize(np.cov( patches[:,i,j,:,:].reshape(n_channels, wt) ))
                                         for j in range(ps[2]) ] for i in range(ps[1]) ]
            _num_corrector = 1.0
            local_covar_img = np.log(np.array(covariance_mat_dets) + _num_corrector)
            if (show_loccov_image and is_color):
                h, w, c = img.shape
                resized_local_covar_img = skimage.transform.resize(local_covar_img, output_shape=(h,w), order=3)
                imdisplay(resized_local_covar_img, 'Local Covariances', cmap = 'viridis', 
                          colorbar = True, mask = img_mask) 
            return np.mean(local_covar_img)
        add_new(local_patch_covariance(img), 3)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #>> Measure 4: Average gradient magnitude of the input
    # Note: this computes the second-order derivatives if we're using a gradient image
    # Note: even if we mask the gradient image, the gradient on the boundary will still be high
    if 4 in complexities_to_use:
        @timing_decorator(args.timing)
        def avg_gradient_norm(img):
            if verbose: print('Computing average edginess')
            grad_img = generate_gradient_magnitude_image(img)
            if (show_gradient_img and is_color):
                gi_title = "Mean Gradient Magnitude Image" + (" (2nd order)" if use_gradient_image else "")
                imdisplay(grad_img.mean(axis=2), gi_title, cmap='plasma', colorbar=True, mask = img_mask)
            if using_alpha_mask:
                grad_img[img_mask <= 0] = 0
            return np.mean(grad_img)
        add_new(avg_gradient_norm(img), 4)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #>> Measure 5: Continuous-space Differential Shannon entropy across pixels
    if 5 in complexities_to_use:
        @timing_decorator(args.timing)
        def cont_pixelwise_diff_ent(img):
            if verbose: print('Computing continuous pixel-wise differential entropy')
            float_img = skimage.img_as_float(img).reshape(h * w, n_channels) # list of pixel values
            if using_alpha_mask:
                unfolded_mask = img_mask.reshape(h * w)
                _str_tmp = "\tMask Sum: " + str(unfolded_mask.sum()) + ", Orig Shape: " + str(float_img.shape)
                float_img = float_img[ unfolded_mask > 0 ]
                _str_tmp += ", Masked Shape: " + str(float_img.shape)
                if verbose: print(_str_tmp)
            # Compute nearest neighbour-based density estimator
            pixelwise_diff_entropy = EE_cont.entropy(float_img, base=np.e)
            if transform_diff_ent: pixelwise_diff_entropy = _oneparam_affine(pixelwise_diff_entropy, affine_param)
            return pixelwise_diff_entropy
        add_new(cont_pixelwise_diff_ent(img), 5)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #>> Measure 6: Continuous-space Differential Shannon entropy over patches
    # Note 1: this measure is not invariant to rotations of patches
    # Note 2: for images with large swathes of identical patches, this method can suffer large increases
    #   in computational cost (due to KD-tree construction/querying difficulties I suspect).
    if 6 in complexities_to_use:
        @timing_decorator(args.timing)
        def patchwise_diff_ent(img):
            if verbose: print('Computing continuous patch-wise differential entropy')
            cimg = skimage.img_as_float(img) # + 1e-6 * np.random.randn(*img.shape)
            # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
            patches_dse, ps, wt = patches_over_channels(cimg, diff_ent_patch_size, diff_ent_window_step, floatify=False)
            if verbose: print('\tPatch windows size:', patches_dse.shape)
            # Gathers unfolded patches across the image
            if using_alpha_mask:
                alpha_over_patches = patches_per_channel(img_mask, diff_ent_patch_size, diff_ent_window_step)
                patch_vectors = vectorize_masked_patches(patches_dse, alpha_over_patches, ps[1], ps[2])
            else:
                patch_vectors = np.array([ patches_dse[:,i,j,:,:].reshape(wt * n_channels) for i in range(ps[1]) for j in range(ps[2]) ])
            # Randomly resample patches to meet maximum number present
            if max_patches_cont_DE < patch_vectors.shape[0]:
                if verbose: print('\tResampling patch vectors (original size: %s)' % str(patch_vectors.shape))
                new_inds = np.random.choice(patch_vectors.shape[0], size=max_patches_cont_DE, replace=False)
                patch_vectors = patch_vectors[new_inds]
            if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
            # Compute nearest neighbour-based density estimator
            patchwise_diff_entropy = EE_cont.entropy(patch_vectors, base=np.e)
            if transform_diff_ent: patchwise_diff_entropy = _oneparam_affine(patchwise_diff_entropy, affine_param)
            return patchwise_diff_entropy
        add_new(patchwise_diff_ent(img), 6)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #>> Measure 7: Global patch-wise covariance logdet
    # Note: this measure is also sensitive to patch orientation (e.g., rotating a patch will affect it)
    if 7 in complexities_to_use:
        @timing_decorator(args.timing)
        def patchwise_global_covar(img):
            if verbose: print('Computing global patch-wise covariance')
            # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
            # I.e., C x N_patches_H x N_patches_V x Window_size_H x Window_size_V
            patchesgc, ps, wt = patches_over_channels(img, global_cov_window_size, global_cov_window_step, floatify=global_cov_norm_img)
            if verbose: print('\tPatch windows size:', patchesgc.shape)
            # Gathers unfolded patches across the image
            if using_alpha_mask:
                alpha_over_patchesgc = patches_per_channel(img_mask, global_cov_window_size, global_cov_window_step)
                patch_vectors = vectorize_masked_patches(patchesgc, alpha_over_patchesgc, ps[1], ps[2])
            else:
                patch_vectors = np.array([ patchesgc[:,i,j,:,:].reshape(wt * n_channels) for i in range(ps[1]) for j in range(ps[2]) ])
            if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
            # Compute covariance matrix of the unfolded vectors
            global_cov = np.cov( patch_vectors.T )
            def _strace(c):
                t = np.trace(c)
                return np.sign(t), t
            scalarizer = (lambda c: np.linalg.slogdet(c)) if not is_scalar else (lambda c: _strace(c))
            sign, patchwise_covar_logdet = scalarizer(global_cov) 
            #patchwise_covar_logdet = np.log( np.linalg.det(global_cov) + 1.0 )
            if global_cov_aff_trans: patchwise_covar_logdet = _oneparam_affine(patchwise_covar_logdet, global_cov_affine_prm)
            return patchwise_covar_logdet
        add_new(patchwise_global_covar(img), 7)

    #--------------------------------------------------------------------------------------------------------------------#
    
    #> Measure 8: Pairwise patch EMD (i.e., mean pairwise Wasserstein distance over patches)
    # TODO bugged? doesnt remove alpha masked patches
    if 8 in complexities_to_use:
        import ot
        @timing_decorator(args.timing)  
        def pairwise_wasserstein_distance(img, use_sinkhorn, sinkhorn_gamma, coordinate_aware, 
                metric, image_rescaling_factor, coordinate_scale, emd_visualize):
            assert metric in ['euclidean', 'sqeuclidean'], "Underlying metric must be 'euclidean' or 'sqeuclidean'"
            if verbose: print('\tImage dims', img.shape)
            # Resize image
            img = skimage.transform.rescale(img, scale=image_rescaling_factor, anti_aliasing=True, multichannel=True)
            if verbose: print('\tDownscaled image dims:', img.shape)
            if (emd_visualize and is_color): imdisplay(img, 'Downscaled img')
            # Extract patches
            patches_emd, ps, wt = patches_over_channels(img, emd_window_size, emd_window_step, floatify=True)
            if verbose: print('\tPatches Shape', patches_emd.shape)
            # CASE 1: using alpha mask
            if using_alpha_mask:
                if (emd_visualize and is_color): imdisplay( img_mask, 'Original Mask', True )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img_mask_ds = skimage.transform.rescale( 
                                            skimage.img_as_float(skimage.img_as_ubyte(img_mask)), 
                                            scale=image_rescaling_factor)
                img_mask_ds[ img_mask_ds >  1e-8 ] = 1
                img_mask_ds[ img_mask_ds <= 1e-8 ] = 0
                if (emd_visualize and is_color): imdisplay(img_mask_ds, 'Downscaled mask', True)
                alpha_over_patches_emd = patches_per_channel(img_mask_ds, emd_window_size, emd_window_step)
                patch_lists = vectorize_masked_patches(patches_emd, alpha_over_patches_emd, ps[1], ps[2], as_list=True)
            # CASE 2: no alpha channel
            else:
                patch_lists = np.array([ patches_emd[:,i,j,:,:].reshape(n_channels, wt).T 
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
                # Index into list (N_patches x unfolded_window_size x n_channels)
                patches_to_vis = patch_lists[ris, :, :].reshape(n_to_vis, emd_window_size, emd_window_size, n_channels)
                patch_display(patches_to_vis, nr, nc, show=False, title='EMD Patches', subtitles=ris)
            # Append coordinates if desired
            if coordinate_aware:
                if verbose: print('\tUsing coordinate-aware calculations')
                n_patches = patch_lists.shape[0]
                # Refold vectorized patches back into windows
                refolded_patches = patch_lists.reshape(n_patches, emd_window_size, emd_window_size, n_channels)
                # Linear walk from 0 to 1
                lin_walk = np.linspace(0.0, 1.0, emd_window_size)
                # Expand linear walk to normalized patch coordinates, and rescale spatial values
                patch_of_coords = np.transpose( np.array( np.meshgrid(lin_walk, lin_walk) ), (1,2,0) ) * coordinate_scale
                # Duplicate patches for each image patch
                coord_patches = np.repeat( patch_of_coords[ np.newaxis, :, : ], n_patches, axis=0 )
                # Concatenate image patch pixel values with coordinate values
                patch_lists = np.concatenate( (coord_patches, refolded_patches), axis=3 ).reshape(n_patches, wt, coord_patches.shape[-1]+n_channels)
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

        # Controls scale (and thus relative weight) of coordinate vs pixel distance.
        # By default, patch pixels are given local x,y, coords in [0,alpha], where alpha
        # is the coordinate scale. If alpha is too large, pixel space distance is ignored;
        # too small, intra-patch spatial distance is ignored.
        if verbose: print('Computing mean inter-patch pairwise Wasserstein distance')
        emd_resize = 1.0 if running_resize else args.emd_downscaling
        emd_args = { 'use_sinkhorn'           : args.sinkhorn_emd,
                     'sinkhorn_gamma'         : args.sinkhorn_regularizer,
                     'coordinate_scale'       : args.emd_coord_scaling, 
                     'coordinate_aware'       : not args.emd_ignore_coords,
                     'image_rescaling_factor' : emd_resize,
                     'emd_visualize'          : args.emd_visualize,
                     'metric'                 : 'sqeuclidean' if args.squared_euc_metric else 'euclidean' }
        if verbose: print('\tParams', emd_args)
        add_new(pairwise_wasserstein_distance(img, **emd_args), 8)

    #--------------------------------------------------------------------------------------------------------------------#    
    
    # HACK 1: Mask handling for measures > 9
    if using_alpha_mask:
        mask_pwmm = img_mask
    else:
        mask_pwmm = np.ones((h,w), dtype=int)

    # HACK 2: pw-mnt discretization parameter
    pw_mnt_dist_nonOL_WS = list(map(int, args.pw_mnt_dist_nonOL_WS.strip().split(",")))
    
    # TODO ok for scalar images?

    # Measure 9: mean-matching pairwise distances
    if 9 in complexities_to_use:
        if verbose: print('Computing patch-wise distances between means')
        @timing_decorator(args.timing)
        def patchwise_mean_dist(img, mask):
            return pairwise_moment_distances(img, mask, block_cuts=pw_mnt_dist_nonOL_WS,
                    gamma_mu_weight=1.0,
                    gamma_cov_weight=0.0, 
                    display_intermeds=(show_pw_mnt_ptchs and is_color), verbose=verbose)
        add_new(patchwise_mean_dist(img, mask_pwmm), 9)

    #--------------------------------------------------------------------------------------------------------------------#
    
    # Measure 10: moment-matching pairwise distances
    if 10 in complexities_to_use:
        # Don't display the same patches twice
        if 9 in complexities_to_use: 
            show_pw_mnt_ptchs2 = False
        else:
            show_pw_mnt_ptchs2 = show_pw_mnt_ptchs
        if verbose: print('Computing patch-wise distances between central moments')
        @timing_decorator(args.timing)
        def patchwise_moment_dist(img, mask):
            return pairwise_moment_distances(img, mask, block_cuts=pw_mnt_dist_nonOL_WS, 
                    gamma_mu_weight=args.gamma_mu_weight,
                    gamma_cov_weight=args.gamma_cov_weight, 
                    display_intermeds=(show_pw_mnt_ptchs2 and is_color), verbose=verbose)
        add_new(patchwise_moment_dist(img, mask_pwmm), 10)

    #--------------------------------------------------------------------------------------------------------------------#
    
    # Measure 11: absolute sum of discrete wavelet transform detail coefficients
    if 11 in complexities_to_use:
        if verbose: print('Computing DWT-based measure')
        from dwtComplexityScore import evalComplexity as get_dwt_complexity, visualize as vis_dwt
        @timing_decorator(args.timing)
        def dwt_complexity_handler(img, mask):
            if show_dwt:
                vis_dwt(img, mask, levels=args.wt_n_levels, mWavelet=args.wt_mother_wavelet, show=False)
            return get_dwt_complexity(img, mask, levels=args.wt_n_levels, thrPercentile=args.wt_threshold_percentile,
                        mWavelet=args.wt_mother_wavelet)
        add_new(dwt_complexity_handler(img, mask_pwmm), 11)

    #--------------------------------------------------------------------------------------------------------------------#
    
    # Measure 12: Pairwise distributional patch matching - Symmetric KL divergence (Jeffrey's divergence)
    # Measure 13: Pairwise distributional patch matching - Wasserstein-2 metric
    # Measure 14: Pairwise distributional patch matching - Hellinger distance
    # Measure 15: Pairwise distributional patch matching - Bhattacharyya distance
    # Measure 16: Pairwise distributional patch matching - Mean-normed FM-eigendistance (FM-ATF)
    # Based on the Forstner-Moonen metric on covariance matrices and the Bhattacharyya-normalized
    #   mean-matching term, similar to Abou-Moustafa and Ferrie, 2012.
    _metric_dict = { 12 : 'pw_symmetric_KL',  13 : 'pw_W2', 14 : 'pw_Hellinger',
                     15 : 'pw_bhattacharyya', 16 : 'pw_FMATF' }
    _set_shown_pw = False
    for _p_com in [12, 13, 14, 15, 16]:
        if _p_com in complexities_to_use:
            if verbose: print('Computing pairwise distributional patch-matching distance', _metric_dict[_p_com])
            # If we were going to show the patches, we did so in 9 (if it was specified) or 10.
            if 9 in complexities_to_use or 10 in complexities_to_use: 
                show_pw_mnt_ptchs_curr = False
            else:
                if not _set_shown_pw: # we haven't asked to show yet
                    show_pw_mnt_ptchs_curr = show_pw_mnt_ptchs
                    # Now that we have specified to show the pw moment patches (if it were asked for), we must ensure
                    # that next targets in the loop do not also do so
                    _set_shown_pw = True 
                else: # set_show was true -> we've already seen it
                    show_pw_mnt_ptchs_curr = False
            # Define and run the current PW matcher
            @timing_decorator(args.timing)
            def patchwise_moment_dist_c(img, mask):
                return pairwise_moment_distances(img, mask, 
                        block_cuts        = pw_mnt_dist_nonOL_WS, # TODO option for overlapping as well
                        gamma_mu_weight   = None,
                        gamma_cov_weight  = None,
                        display_intermeds = (show_pw_mnt_ptchs_curr and is_color), 
                        verbose           = verbose,
                        mode              = _metric_dict[_p_com] )
            add_new(patchwise_moment_dist_c(img, mask_pwmm), _p_com)

    ### </ Finished computing complexity measures /> ###

    #######################################################################################################################
    
    # Minor checks
    assert all([v1 == v2 for v1,v2 in zip(S[1:], computed_names)]), 'Mismatch between intended and computed measures'

    # Show images if needed
    if (( show_fourier_image or display_image or show_locent_image or show_loccov_image or show_gradient_img
         or args.emd_visualize or show_pw_mnt_ptchs) and is_color):
        plt.show()

    ### Print (single image case) and return output ###
    complexities_strings = list(map(lambda f: '%.4e' % f, complexities))
    impath = impath + (" [GradMag]" if use_gradient_image else "")
    # Print results for a single image
    if print_mode == 'single':
        print(impath + (" (Gradient Magnitude)" if use_gradient_image else ""))
        if args.timing:
            for name, val, time in zip(S[1:], complexities, timings): print('\t%s: %.4e (time: %.2fs)' % (name,val,time))
            print('Total time elapsed:', sum(timings))
        else:
            for name, val in zip(S[1:], complexities): print('\t%s: %.4e' % (name,val))
    # Print results in the multi-image case
    elif print_mode == 'compact':
        print("%s,%s" % (impath,",".join(complexities_strings)))
        if args.timing:
            print('  |> Timings:', ",".join( list(map(lambda f: '%.3f' % f, timings)) ), '[Total: %.1fs]' % sum(timings))
    # Return final output
    if args.timing: return impath, complexities_strings, timings    
    return impath, complexities_strings
    # </ End of single image complexity calculation /> #

#------------------------------------------------------------------------------#

### Arguments based on user input ###
path = args.input # Path to target
args_d = { "verbose"            : args.verbose,
           "is_mspec"           : args.is_mspec,
           "use_gradient_image" : args.use_grad,
           "show_fourier_image" : True if args.show_all else args.show_fourier,
           "show_gradient_img"  : True if args.show_all else args.show_gradient_img,
           "show_locent_image"  : True if args.show_all else args.show_locents,
           "show_pw_mnt_ptchs"  : True if args.show_all else args.show_pw_mnt_ptchs,
           "show_loccov_image"  : True if args.show_all else args.show_local_covars,
           "show_dwt"           : True if args.show_all else args.show_dwt,
           "display_image"      : True if args.show_all else args.show_img }
grad_and_orig = args.use_grad_too # Preparations if doing both gradient and original image
if grad_and_orig: del args_d['use_gradient_image']

if (args_d['is_mspec']): ### For multispectral images
    
    usables = ['.tif','.txt','.csv']
    usables = list(set( usables + [ b.upper() for b in usables ] + [ b.lower() for b in usables ] ))
    _checker = lambda k: any( k.endswith(yy) for yy in usables )
    
    if os.path.isdir(path):
        ### Case 1: Compute complexities over a folder of images ###
        print(','.join(S))
        for f in [ f for f in os.listdir(path) if _checker(f) ]:
            compute_complexities(os.path.join(path,f), complexities_to_use, print_mode='compact', **args_d)
            if grad_and_orig: # Just did original
                compute_complexities(os.path.join(path,f), complexities_to_use, print_mode='compact',
                                            use_gradient_image=True, **args_d)
    elif (path[-4:] == ".csv"):
        im_stack_filenames = load_csv_data(path)
        
        for im_stack in im_stack_filenames:
            compute_complexities(im_stack, complexities_to_use, print_mode='compact', **args_d)
            if grad_and_orig:
                compute_complexities(im_stack, complexities_to_use, print_mode='compact', use_gradient_image=True, **args_d)
    else:
        ### Case 2: Compute complexity measure on a single image ###
        if (not _checker(path)):
            raise TypeError('Image format is not supported for multispectral image processing.')
            
        compute_complexities(path, complexities_to_use, print_mode='single', **args_d)
        if grad_and_orig:
            compute_complexities(path, complexities_to_use, print_mode='single', use_gradient_image=True, **args_d)
            
else: ### For color images
   
    usables = [ '.jpg', '.png' ]
    usables = list(set( usables + [ b.upper() for b in usables ] + [ b.lower() for b in usables ] ))
    _checker = lambda k: any( k.endswith(yy) for yy in usables )
    
    if os.path.isdir(path):
         ### Case 1: Compute complexities over a folder of images ###
        print(','.join(S))
        for f in [ f for f in os.listdir(path) if _checker(f) ]:
            compute_complexities(os.path.join(path,f), complexities_to_use, print_mode='compact', **args_d)
            if grad_and_orig: # Just did original
                compute_complexities(os.path.join(path,f), complexities_to_use, print_mode='compact',
                                            use_gradient_image=True, **args_d)
    else:
        ### Case 2: Compute complexity measure on a single image ###
        if (not _checker(path)):
            raise TypeError('Image format is not supported for color image processing.')
            
        compute_complexities(path, complexities_to_use, print_mode='single', **args_d)
        if grad_and_orig:
            compute_complexities(path, complexities_to_use, print_mode='single', use_gradient_image=True, **args_d)


#
