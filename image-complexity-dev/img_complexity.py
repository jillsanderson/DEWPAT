import os, sys, numpy as np, skimage, argparse, warnings, random
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from module.complexities import *
from module.utils import *
from pathlib import Path

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
# Preprocessing (resize, blur, greyscale, etc...)
parser.add_argument('--blur', type=float, default=0.0,
    help='Specify Gaussian blur standard deviation applied to the image')
parser.add_argument('--greyscale', type=str, default="none",
    help='Specify greyscale conversion: one of "human", "avg", or "none".')
parser.add_argument('--resize', type=float, default=1.0,
    help='Specify scalar resizing value. E.g., 0.5 halves the image size; 2 doubles it.')
parser.add_argument('--rseed', type=int, default=None,
    help='Specify a fixed random seed to use for this run.')
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
#group_c.add_argument('--pairwise_emd',                  
#        dest='pairwise_emd', action='store_const', 
#        const=8, default=None,
#        help='Whether to use the pairwise Wasserstein distance across image patches')
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
#group_p.add_argument('--sinkhorn_emd', action='store_true', 
#    help='Specify to compute the entropy-regularized Sinkhorn approximation, rather than the exact EMD via linear programming')
#group_p.add_argument('--emd_ignore_coords', action='store_true',
#    help='Specify to avoid appending local normalized spatial coordinates to patch elements')
#group_p.add_argument('--squared_euc_metric', action='store_true',
#    help='Specify to use squared Euclidean rather than Euclidean underlying metric for the EMD')
#group_p.add_argument('--emd_downscaling', type=float, default=0.2,
#    help='Specify image downscaling factor for EMD calculations')
#group_p.add_argument('--sinkhorn_regularizer', type=float, default=0.25,
#    help='Specify Sinkhorn entropy regularization weight coefficient')
#group_p.add_argument('--emd_coord_scaling', type=float, default=0.2,
#    help='Specify spatial coordinate scaling for EMD calculations, which controls the relative balance between pixel vs image space distance')
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
group_p.add_argument('--fourier_wt', type=str, default="l2", choices = ["l1", "l2"],
    help='Weighting scheme for Fourier complexity calculation')

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
#group_v.add_argument('--show_emd_intermeds', 
#        dest='emd_visualize', action='store_true',
#        help='Whether to visualize intermediate computations in the EMD method')
group_v.add_argument('--show_img',          
        dest='show_img', action='store_true',
        help='Whether to display the input image')
group_v.add_argument('--show_dwt',
        dest='show_dwt', action='store_true',
        help='Whether to display the discrete wavelet transform intermediaries')
group_v.add_argument('--show_all',          
        dest='show_all', action='store_true',
        help='Whether to display all of the above images')
group_v.add_argument('--save_vis_to',          
        type = str,
        help='Folder into which to save all viewed visualizations')
group_v.add_argument('--no_display',          
        dest='no_display', action='store_true',
        help='Pass this flag to turn off opening the vis windows (use with save_vis_to)')

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

args.pairwise_emd = None
args.sinkhorn_emd = False
args.emd_ignore_coords = False
args.squared_euc_metric = False
args.emd_downscaling = False
args.sinkhorn_regularizer = 0.25
args.emd_coord_scaling = 0.2
args.show_emd_intermeds = False
args.emd_visualize = False

if not args.rseed is None:
    # Note that this is the legacy approach, not the ideal one, for numpy 
    if args.verbose: print('Setting random seed to', args.rseed)
    np.random.seed(args.rseed)
    random.seed(args.rseed)

if not args.save_vis_to is None:
    if args.verbose: print('Saving visualizations to', args.save_vis_to)
    if not os.path.isdir(args.save_vis_to):
        os.makedirs(args.save_vis_to)
    else:
        if args.verbose: print(args.save_vis_to, 'exists. Files may be overwritten.')

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
               args.pwg_bhattacharyya_div, args.pwg_fmatf_div ]
assert len(S_all) == len(input_vals)
if all(map(lambda k: k is None, input_vals)):
    # No metric specified
    if args.verbose: print('Using all complexity measures')
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
        verbose=False               # Whether to print verbosely when running
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

    # Read original image
    if verbose: print('Reading image:', impath)
    img = skimage.io.imread(impath)
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
        if n_channels == 4:
            alpha_layer = img[:,:,3]
            if False: # Display 
                imdisplay(alpha_layer, 'alpha_layer', colorbar=True, cmap='plasma')
                imdisplay(img[:,:,0:3], 'layers', colorbar=True, cmap='plasma')
                plt.show()
            alpha_layer[ alpha_layer >  128 ] = 255
            alpha_layer[ alpha_layer <= 128 ] = 0
    
    # Handle alpha transparency
    alpha_channel = img[:,:,3] if n_channels == 4 else None
    using_alpha_mask = not (alpha_channel is None)
    if using_alpha_mask:
        img = img[:,:,0:3]
        alpha_mask = np.copy(alpha_channel).astype(int)
        alpha_mask[ alpha_mask <= 0 ] = 0
        alpha_mask[ alpha_mask  > 0 ] = 1
        if False: # Display mask
            imdisplay(alpha_mask, 'alpha_mask', colorbar=True, cmap='plasma')
            plt.show()
    else:
        alpha_mask = None

    # Ignore the alpha channel, if desired
    if args.ignore_alpha:
        using_alpha_mask = False
        alpha_mask, alpha_channel = None, None
    
    # Convert image to greyscale, if desired
    gs_type = args.greyscale.lower()
    assert gs_type in ["none", "human", "avg"], 'Use one of --greyscale none/avg/human'
    if gs_type == 'human':
        if verbose: print("Greyscaling image (perceptual)")
        img = to_perceptual_greyscale(img)
    elif gs_type == 'avg':
        if verbose: print("Greyscaling image (channel mean)")
        img = to_avg_greyscale(img)

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
        img[ alpha_mask <= 0 ] = 0
    if display_image: imdisplay(img, 'Image')
    if verbose:
        print('Image Shape:', img.shape)
        print('Channelwise Min/Max')
        for i in range(3):
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
        add_new(discrete_pixelwise_shannon_entropy(img, alpha_mask=alpha_mask, verbose=verbose, timing=args.timing), 0)

    #--------------------------------------------------------------------------------------------------------------------#

    #>> Measure 1: Averaged channel-wise local entropy
    if 1 in complexities_to_use:
        add_new(channelwise_local_entropies(img, alpha_mask=alpha_mask, show_locent_image=show_locent_image,
                                            local_entropy_disk_size=local_entropy_disk_size, verbose=verbose,
                                            timing=args.timing), 1)

    #--------------------------------------------------------------------------------------------------------------------#

    #>> Measure 2: High frequency content via weighted average
    # Note: masks, even if present, are not used here, since we would be getting irregular domain harmonics rather than Fourier coefs
    # This does mean, however, that background pixels do still participate in the measure (e.g., a white dewlap on a white bg, will
    # incur different frequency effects than on a black bg). 
    if 2 in complexities_to_use:
        add_new(mean_weighted_fourier_coef(img, 
                                           alpha_mask = alpha_mask, 
                                           weight_type = args.fourier_wt,
                                           show_fourier_image = show_fourier_image, 
                                           verbose = verbose, 
                                           timing = args.timing), 2)

    #--------------------------------------------------------------------------------------------------------------------#

    #>> Measure 3: Local (intra-)patch covariances
    # Closely related to the distribution of L_2 distance between patches
    # Note: vectorize over the patches? use slogdet
    if 3 in complexities_to_use:
        local_patch_size = args.local_cov_patch_size
        local_covar_wstep = args.local_covar_wstep
        add_new(local_patch_covariance(img, alpha_mask=alpha_mask, local_patch_size=local_patch_size,
                                       local_covar_wstep=local_covar_wstep, show_loccov_image=show_loccov_image,
                                       verbose=verbose, timing=args.timing), 3)

    #--------------------------------------------------------------------------------------------------------------------#

    #>> Measure 4: Average gradient magnitude of the input
    # Note: this computes the second-order derivatives if we're using a gradient image
    # Note: even if we mask the gradient image, the gradient on the boundary will still be high
    if 4 in complexities_to_use:
        add_new(avg_gradient_norm(img, alpha_mask=alpha_mask, use_gradient_image=use_gradient_image,
                                  show_gradient_img=show_gradient_img, verbose=verbose, timing=args.timing), 4)

    #--------------------------------------------------------------------------------------------------------------------#

    #>> Measure 5: Continuous-space Differential Shannon entropy across pixels
    if 5 in complexities_to_use:
        add_new(cont_pixelwise_diff_ent(img, alpha_mask=alpha_mask, transform_diff_ent=transform_diff_ent,
                                        affine_param=affine_param, verbose=verbose, timing=args.timing), 5)
 
    #--------------------------------------------------------------------------------------------------------------------#

    #>> Measure 6: Continuous-space Differential Shannon entropy over patches
    # Note 1: this measure is not invariant to rotations of patches
    # Note 2: for images with large swathes of identical patches, this method can suffer large increases
    #   in computational cost (due to KD-tree construction/querying difficulties I suspect).
    if 6 in complexities_to_use:
        add_new(patchwise_diff_ent(img, alpha_mask=alpha_mask, diff_ent_patch_size=diff_ent_patch_size,
                                   diff_ent_window_step=diff_ent_window_step, max_patches_cont_DE=max_patches_cont_DE,
                                   affine_param=affine_param, transform_diff_ent=transform_diff_ent, verbose=verbose,
                                   timing=args.timing), 6)

    #--------------------------------------------------------------------------------------------------------------------#

    #>> Measure 7: Global patch-wise covariance logdet
    # Note: this measure is also sensitive to patch orientation (e.g., rotating a patch will affect it)
    if 7 in complexities_to_use:
        add_new(patchwise_global_covar(img, alpha_mask=alpha_mask, global_cov_window_size=global_cov_window_size,
                                       global_cov_window_step=global_cov_window_step, global_cov_norm_img=global_cov_norm_img,
                                       global_cov_aff_trans=global_cov_aff_trans,
                                       global_cov_affine_prm=global_cov_affine_prm, verbose=verbose, timing=args.timing), 7)

    #--------------------------------------------------------------------------------------------------------------------#

    #> Measure 8: Pairwise patch EMD (i.e., mean pairwise Wasserstein distance over patches)
    # TODO bugged? doesnt remove alpha masked patches
    if 8 in complexities_to_use:

        # Controls scale (and thus relative weight) of coordinate vs pixel distance.
        # By default, patch pixels are given local x,y, coords in [0,alpha], where alpha
        # is the coordinate scale. If alpha is too large, pixel space distance is ignored;
        # too small, intra-patch spatial distance is ignored.
        if verbose: print('Computing mean inter-patch pairwise Wasserstein distance')
        # --resize overrides the value of --emd_downscaling
        emd_resize = 1.0 if running_resize else args.emd_downscaling
        emd_args = { 'use_sinkhorn'           : args.sinkhorn_emd,
                     'sinkhorn_gamma'         : args.sinkhorn_regularizer,
                     'coordinate_scale'       : args.emd_coord_scaling, 
                     'coordinate_aware'       : not args.emd_ignore_coords,
                     'image_rescaling_factor' : emd_resize,
                     'emd_visualize'          : args.emd_visualize,
                     'metric'                 : 'sqeuclidean' if args.squared_euc_metric else 'euclidean' }
        if verbose: print('\tParams', emd_args)
        add_new(pairwise_wasserstein_distance(img, **emd_args, alpha_mask=alpha_mask, emd_window_size=emd_window_size,
                                              emd_window_step=emd_window_step, verbose=verbose, timing=args.timing), 8)

    #--------------------------------------------------------------------------------------------------------------------#    

    # HACK 1: Mask handling for measures > 9
    if using_alpha_mask:
        mask_pwmm = alpha_mask
    else:
        mask_pwmm = np.ones((h,w), dtype=int)

    # HACK 2: pw-mnt discretization parameter
    pw_mnt_dist_nonOL_WS = list(map(int, args.pw_mnt_dist_nonOL_WS.strip().split(",")))
    
    # TODO ok for scalar images?

    # Measure 9: mean-matching pairwise distances
    if 9 in complexities_to_use:
        if verbose: print('Computing patch-wise distances between means')
        add_new(patchwise_mean_dist(img, mask_pwmm, pw_mnt_dist_nonOL_WS=pw_mnt_dist_nonOL_WS,
                                    show_pw_mnt_ptchs=show_pw_mnt_ptchs, verbose=verbose, timing=args.timing), 9)

    #--------------------------------------------------------------------------------------------------------------------#

    # Measure 10: moment-matching pairwise distances
    if 10 in complexities_to_use:
        # Don't display the same patches twice
        if 9 in complexities_to_use: 
            show_pw_mnt_ptchs2 = False
        else:
            show_pw_mnt_ptchs2 = show_pw_mnt_ptchs
        if verbose: print('Computing patch-wise distances between central moments')
        add_new(patchwise_moment_dist(img, mask_pwmm, pw_mnt_dist_nonOL_WS=pw_mnt_dist_nonOL_WS,
                                      gamma_mu_weight=args.gamma_mu_weight, show_pw_mnt_ptchs=show_pw_mnt_ptchs2,
                                      gamma_cov_weight=args.gamma_cov_weight, verbose=verbose, timing=args.timing), 10)

    #--------------------------------------------------------------------------------------------------------------------#
    
    # Measure 11: absolute sum of discrete wavelet transform detail coefficients
    if 11 in complexities_to_use:
        if verbose: print('Computing DWT-based measure')
        from module.complexities import evalComplexity as get_dwt_complexity, visualize as vis_dwt
        @timing_decorator()
        def dwt_complexity_handler(img, mask, timing=args.timing):
            if show_dwt:
                vis_dwt(img, mask, levels=args.wt_n_levels, mWavelet=args.wt_mother_wavelet, show=False)
            return get_dwt_complexity(img, mask, levels=args.wt_n_levels, thrPercentile=args.wt_threshold_percentile,
                        mWavelet=args.wt_mother_wavelet)
        add_new(dwt_complexity_handler(img, mask_pwmm, timing=args.timing), 11)

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
            add_new(patchwise_moment_dist_c(img, mask_pwmm, _metric_dict[_p_com], pw_mnt_dist_nonOL_WS=pw_mnt_dist_nonOL_WS,
                                            show_pw_mnt_ptchs=show_pw_mnt_ptchs_curr, verbose=verbose, timing=args.timing), _p_com)

    ### </ Finished computing complexity measures /> ###

    #######################################################################################################################

    # Minor checks
    assert all([v1 == v2 for v1,v2 in zip(S[1:], computed_names)]), 'Mismatch between intended and computed measures'

    # Show images if needed
    if ( show_fourier_image or display_image or show_locent_image or show_loccov_image or show_gradient_img
         or args.emd_visualize or show_pw_mnt_ptchs or show_dwt):
        # Save displayed figures to given folder
        if not args.save_vis_to is None:
            # Get filename
            img_filename_s = Path(impath).stem.strip().replace(" ", "_").lower()
            if args.verbose: print('Saving visualizations for', img_filename_s, 'to', args.save_vis_to)
            # Save figures
            for fign in plt.get_fignums():
                # Try: suptitle -> axis 1 title
                caxes = plt.figure(fign).axes
                stitle = plt.figure(fign)._suptitle
                if stitle is None:
                    if len(caxes) == 0:
                        subname = 'subfig' 
                    else:
                        subname = plt.figure(fign).axes[0].get_title().strip().replace(" ", "_").lower()
                else:
                    subname = stitle.get_text().strip().replace(" ", "_").lower()
                subname = subname.replace("(", "").replace(")", "")
                p2save  = os.path.join(args.save_vis_to, img_filename_s + "." + subname + ".png")
                plt.figure(fign).savefig(p2save)
        # Open up actual display
        if not args.no_display:
            plt.show()

    ### Print (single image case) and return output ###
    complexities_strings = list(map(lambda f: '%.4f' % f, complexities))
    impath = impath + (" [GradMag]" if use_gradient_image else "")
    # Print results for a single image
    if print_mode == 'single':
        print(impath + (" (Gradient Magnitude)" if use_gradient_image else ""))
        if args.timing:
            for name, val, time in zip(S[1:], complexities, timings): print('\t%s: %.4f (time: %.2fs)' % (name,val,time))
            print('Total time elapsed:', sum(timings))
        else:
            for name, val in zip(S[1:], complexities): print('\t%s: %.4f' % (name,val))
    # Print results in the multi-image case
    elif print_mode == 'compact':
        print("%s,%s" % (impath,",".join(complexities_strings)))
        if args.timing:
            print('  |> Timings:', ",".join( list(map(lambda f: '%.3f' % f, timings)) ), '[Total: %.1fs]' % sum(timings))
    # Clean up visualizations 
    plt.close('all')
    # Return final output
    if args.timing: return impath, complexities_strings, timings    
    return impath, complexities_strings
    # </ End of single image complexity calculation /> #

#------------------------------------------------------------------------------#

### Arguments based on user input ###
path = args.input # Path to target
args_d = { "verbose"            : args.verbose,
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

### Case 1: Compute complexities over a folder of images ###
# Meant to output a CSV
if os.path.isdir(path):
    print(','.join(S))
    usables = [ '.jpg', '.png' ]
    usables = list(set( usables + [ b.upper() for b in usables ] + [ b.lower() for b in usables ] ))
    _checker = lambda k: any( k.endswith(yy) for yy in usables )
    for f in [ f for f in os.listdir(path) if _checker(f) ]:
        compute_complexities(os.path.join(path,f), complexities_to_use, print_mode='compact', **args_d)
        if grad_and_orig: # Just did original
            compute_complexities(os.path.join(path,f), complexities_to_use, print_mode='compact',
                                        use_gradient_image=True, **args_d)
### Case 2: Compute complexity measure on a single image ###
else: # Single image case
    compute_complexities(path, complexities_to_use, print_mode='single', **args_d)
    if grad_and_orig:
        compute_complexities(path, complexities_to_use, print_mode='single', use_gradient_image=True, **args_d)


#
