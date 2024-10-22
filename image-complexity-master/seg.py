import os, sys, utils, argparse, sklearn, skimage, numpy as np, numpy.random as npr
import warnings, matplotlib.pyplot as plt, csv
from skimage import data, segmentation, color
from skimage.color import rgb2hsv
from skimage.future import graph
from sklearn import cluster
from scipy import interpolate
from collections import Counter
from utils import *

_glob_form = Formatter()

# TODO save json of counters and matrices for folders

def main():
    ### >> Argument parsing << ###
    parser = argparse.ArgumentParser(
                description='Image clustering/segmentation analysis.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument('input', type=str, help='Input: either a folder or an image')
    parser.add_argument('--labeller', dest='labeller', type=str, default='kmeans',
        choices=['affinity_prop', 'dbscan', 'optics', 'graph_cuts', 'kmeans'],
        help='Labelling (segmentation and/or clustering) algorithm.')
    parser.add_argument('--verbose', dest='verbose', default=False, action='store_true',
        help='Whether to print verbosely while running')
    parser.add_argument('--mspec', dest='is_mspec', action='store_true',
        help='Whether to treat the input as a multispectral image')
    ### Image preprocessing
    parser.add_argument('--resize', type=float, default=0.5,
        help='Specify scalar resizing. E.g., 0.5 halves the image size; 2 doubles it. (default: 0.5)')
    parser.add_argument('--blur', type=float, default=1.0,
        help='Specify Gaussian blur standard deviation applied to the image (default: 1)')
    parser.add_argument('--ignore_alpha', action='store_true', 
        help='Pass to ask the algorithm to ignore the alpha channel (default: False)')
    parser.add_argument('--clustering_colour_space', default='rgb',
        choices=['rgb', 'hsv', 'cie', 'lab'],
        help='Specify the colour space in which to perform the clustering')
    ### Post-processing options
    parser.add_argument('--merge_small_clusters_method', default='none',
        choices = ['none', 'k_dependent', 'fixed'],
        help='Specify how or whether to merge small clusters')
    parser.add_argument('--fixed_cluster_size_merging_threshold', default=0.05, type=float,
        help='Fixed threshold percentage for cluster merging (only used when merging method is fixed)')
    parser.add_argument('--small_cluster_merging_kdep_param', default=0.05, type=float,
        help='Parameter for k_dependent annealed thresholding (initial thresh for k=2)')
    parser.add_argument('--small_cluster_merging_dynamic_k', action='store_true',
        help='If using k_dependent merging threshold, specify to recompute the threshold after every merge')
    ### Output statistics files
    parser.add_argument('--seg_stats_output_file', default=None,
        help='Specifies output file to which segment statistics should be written')
    parser.add_argument('--cluster_number_file', default=None,
        help='Specifies output file to which the number of estimated clusters per image should be written')
    ### Output image files
    parser.add_argument('--write_mean_segs', action='store_true', dest='write_mean_segs',
        help='Writes the segmented image(s) with cluster-mean values. Requires mean_seg_output_dir.')
    parser.add_argument('--mean_seg_output_dir', default=None,
        help='Specifies the folder to which saved mean segment images must be written')
    ### Labeller params
    group_g = parser.add_argument_group('Labeller parameters')
    #-- DBSCAN options
    group_g.add_argument('--dbscan_eps', dest='dbscan_eps', type=float, default=1.0,
        help='Epsilon neighbourhood used in DBSCAN.')
    group_g.add_argument('--dbscan_min_neb_size', dest='dbscan_min_neb_size', type=int, default=20,
        help='Min samples needed for core point designation in a neighbourhood. Used in DBSCAN.')
    #-- Graph cuts segmentation options
    group_g.add_argument('--gc_compactness', dest='gc_compactness', type=float, default=10.0,
        help='Graph cuts superpixel compactness.')
    group_g.add_argument('--gc_n_segments', dest='gc_n_segments', type=int, default=300,
        help='Graph cuts n_segments used in initialization.')
    group_g.add_argument('--gc_slic_sigma', dest='gc_slic_sigma', type=float, default=0.0,
        help='Graph cuts Gaussian kernel width for superpixel initialization.')    
    #-- Kmeans options
    group_g.add_argument('--kmeans_k', dest='kmeans_k', default=None,
        help='Specify number of clusters for K-means. Chosen automatically by default.')    
    group_g.add_argument('--kmeans_auto_bounds', dest='kmeans_auto_bounds', type=str, default="2,6",
        help='Upper and lower bounds on the number of clusters searched over for kmeans. (e.g., 2,6)')    
    group_g.add_argument('--kmeans_auto_crit', dest='kmeans_auto_crit', type=str, default='davies_bouldin',
        choices=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
        help='Choice of criterion for choosing the best k-value (either mean silhouette coefficient,' 
              'Davies-Bouldin score, or Calinski-Harabasz index).')    
    group_g.add_argument('--kmeans_k_file_list', default=None,
        help='Specifies a path to a csv file that lists k values per image (e.g., "ty.png,5")')
    ### Analysis
    group_a = parser.add_argument_group('Analysis parameters')
    group_a.add_argument('--normalize_matrix', action='store_true',
        help='Whether to normalize the transition matrix')
    group_a.add_argument('--no_print_transitions', action='store_true',
        help='Pass flag to prevent printing the transition matrix')
    group_a.add_argument('--keep_bg', action='store_true',
        help='Keeps the background label when computing the transition matrix')
    ### Visualization
    # TODO heatmap with labels on scale (colorbar), superpixels
    parser.add_argument('--display', dest='display', action='store_true',
        help='Whether to display the resulting labelling')
    args = parser.parse_args()

    ### Checks ###
    # Ensure we have somewhere to write the mean segs, if needed
    if args.write_mean_segs:
        assert not args.mean_seg_output_dir is None, "--write_mean_segs requires --mean_seg_output_dir"
    if not args.mean_seg_output_dir is None: args.write_mean_segs = True
    # Ensure clear choice for k in k means
    _msg_1 = "Specify only one of {kmeans_k, kmeans_k_file_list, kmeans_auto_crit}"
    if not (args.kmeans_k_file_list is None): assert args.kmeans_k is None, _msg_1 
    if not (args.kmeans_k is None): assert args.kmeans_k_file_list is None, _msg_1 

    # Read parameter specifications, if given
    if not (args.kmeans_k_file_list is None):
        assert os.path.isfile(args.kmeans_k_file_list), "K means specifier does not exist"
        if args.verbose: print('Reading kmeans parameter specifier', args.kmeans_k_file_list)
        args.kmeans_specifier = utils.read_csv_full(args.kmeans_k_file_list)
    else:
        args.kmeans_specifier = None
    
    ### Handle images ###
    file_outputs = {}
    if (args.is_mspec): # Multispectral images
        if (False): # Multi image
            raise NotImplementedError('Multi-image multispectral image segmentation is not supported yet.')
        elif os.path.isfile(args.input): # Single image
            usables = ['.tif']
            usables = list(set( usables + [ b.upper() for b in usables ] + [ b.lower() for b in usables ] ))
            _checker = lambda k: any( k.endswith(yy) for yy in usables )
            if (not _checker(args.input)):
                raise TypeError('Image format is not supported for multispectral image processing.')
            file_outputs[args.input] = main_helper(args.input, args)
        else:
            raise ValueError('Non-existent target input ' + args.input)            
    else: # Color images
        usables = [ '.jpg', '.png' ]
        usables = list(set( usables + [ b.upper() for b in usables ] + 
                                      [ b.lower() for b in usables ] ))
        _checker = lambda k: any( k.endswith(yy) for yy in usables )
        if os.path.isdir(args.input): ### Multi-images
            targets = [ os.path.join(args.input, f) 
                        for f in os.listdir(args.input) if _checker(f) ]
            for t in targets: 
                file_outputs[t] = main_helper(t, args)
        elif os.path.isfile(args.input): ### Single image
            if (not _checker(args.input)):
                raise TypeError('Image format is not supported for color image processing.')
            file_outputs[args.input] = main_helper(args.input, args)
                      
        else:
            raise ValueError('Non-existent target input ' + args.input)
        
    # Write seg info results
    if not (args.seg_stats_output_file is None):
        if args.verbose: print('Writing seg file to', args.seg_stats_output_file)
        with open(args.seg_stats_output_file, "w") as _fh:
            #_fh.write("file,cluster_ind,mu_C_1,mu_C_2,mu_C_3,n_member_pixels,percent_member_pixels\n")
            # Check first output dict to see if HSV data is included
            header_line = ['image,cluster,']
            _,testDict = file_outputs[next(iter(file_outputs))]
            D = testDict['cluster_info']['cluster_stats'][next(iter(testDict['cluster_info']['cluster_stats']))]
            mu_keys = [s for s in D.keys() if "mean_value" in s]
            for i in range(len(mu_keys)):
                header_line.append('mu_ch'+str(i+1) + ',')
            if (not args.is_mspec):
                header_line.append('mu_H,mu_S,mu_V,')
            header_line.append('frequency,percent\n')
            _fh.write(''.join(header_line))

            for key in file_outputs.keys(): # For each file
                tmat, D_img = file_outputs[key]
                Ds = D_img['cluster_info']['cluster_stats']
                clines = []
                for cluster_label in Ds.keys(): # For each cluster
                    D = Ds[cluster_label]
                    cline = [key, D['label_number']]
                    mu_keys = [s for s in D.keys() if "mean_value" in s]
                    for i in range(len(mu_keys)):
                        cline.append(D[mu_keys[i]])
                    if (not args.is_mspec):
                        cline.extend([D['mean_colour_H'], D['mean_colour_S'], D['mean_colour_V']])
                    cline.extend([D['n_member_pixels'], D['percent_member_pixels']])
                    clines.append(cline)
                clines.sort(key = lambda a: a[-1], reverse = True)
                clines = [ ",".join( map(str, a) ) for a in clines ]
                for line in clines: _fh.write(line + '\n')
    # Write csv with number of clusters per image
    # The background label is NOT counted
    if not (args.cluster_number_file is None):
        if args.verbose: print('Writing cluster number file to', args.cluster_number_file)
        with open(args.cluster_number_file, 'w') as fh:
            fh.write("image,number_of_clusters\n")
            for key in file_outputs.keys():
                tmat, D_img = file_outputs[key]
                allowed_labels = D_img['cluster_info']['allowed_labels']
                num_labels = len(allowed_labels)
                if -1 in allowed_labels: num_labels -= 1
                fh.write( "%s,%d\n" % (key,num_labels) )
    
def main_helper(img_path, args):
    #img_filename_extless, img_file_extension = os.path.splitext(img_path)
    img_file_basename = os.path.basename(img_path)
    img_file_basename_extless, img_file_ext = os.path.splitext(img_file_basename)

    kmeans_specifier = args.kmeans_specifier
    if not (kmeans_specifier is None):
        targ_row = utils.get_row_via(kmeans_specifier, img_file_basename, 0)
        if targ_row is None:
            print("Error: unable to find", img_file_basename, "in k-spec file! Skipping target.")
            return
        else:
            found_k = int(kmeans_specifier[targ_row][1].strip())
            if args.verbose: 
                print('Found k-means spec target %s at row %d' % (img_file_basename, targ_row))
                print('\tObtained k-value of', found_k)
            args.kmeans_k = found_k

    if (not args.is_mspec):
        img, img_mask = load_color_image(img_path)
    else:
        img, img_mask = load_mspec_image(img_path)
        raise NotImplementedError('Multispectral images cannot be loaded in segmentation module yet.')
        
    if args.verbose: print('Loaded image', img_path)
    n_channels = img.shape[2]

    # Downscale image, if desired
    resize_factor_main = args.resize
    assert resize_factor_main > 0.0, "--resize must be positive"
    running_resize = False
    if abs(resize_factor_main - 1.0) > 1e-4:
        running_resize = True
        if args.verbose: print("Orig shape", img.shape, "| Resizing by", resize_factor_main)
        img = skimage.transform.rescale(img, scale=resize_factor_main,
                anti_aliasing=True, multichannel=True)
        img = utils.conv_to_ubyte(img)
        if (img_mask is not None):
            img_mask = skimage.transform.rescale(img_mask, scale=resize_factor_main,
                anti_aliasing=True, multichannel=False)
            img_mask = utils.conv_to_ubyte(img_mask)
            img_mask[ img_mask > 0] = 1
            img_mask[ img_mask <= 0] = 0
        if args.verbose: print("Resized dims:", img.shape)
    
    # Handle masking (alpha transparency)
    if (img_mask is not None):
        # Zero out the background
        if not args.ignore_alpha:
            img[img_mask <= 0] = 0
        
    # Default mask: non-transparent alpha channel
    H, W,_  = img.shape
    if (img_mask is None) or args.ignore_alpha: 
        img_mask = np.ones( (H,W) )
    
    # Gaussian filter if desired
    if args.blur > 1e-8:
        if args.verbose: print("\tBlurring with sigma =", args.blur)
        img = gauss_filter(img, args.blur)
        
    if args.verbose: 
        print('\tShape (%d,%d,%d)\n\tmin/max vals:' % (H,W,n_channels), 
              img.min(0).min(0), img.max(0).max(0))
        print('\tNum masked values:', H*W - (img_mask > 0).sum(), "/", H*W)
    
    ### Label (cluster/segment) the image ###
    def reorder_label_img(LI):
        if args.verbose: print('\tReordering label image')
        LI = np.rint(LI).astype(int)
        labels_list = LI.flatten().astype(int).tolist()
        c = Counter(labels_list)
        if args.verbose: print('\tOrig Counter:', c)
        ordered_targets = c.most_common() # [(int_key, count),...]
        curr_new_label = 1
        LI_new = np.copy(LI)
        for j, (int_key, count) in enumerate(ordered_targets):
            if int_key == -1: continue # Leave the background alone
            LI_new[ LI == int_key ] = curr_new_label
            curr_new_label += 1
        if args.verbose: print('\tNew Counter:', Counter(LI_new.flatten().astype(int).tolist()))
        return LI_new
    def k_dependent_smaller_clusters_merging_threshold(k):
        # Start at some value for k=2, and anneal down as k increases
        if k <= 1: return 0.0 # No merging if k == 1
        T0 = args.small_cluster_merging_kdep_param
        T  = T0 / (k - 1)
        return T
    def merge_smaller_clusters(LI):
        LI = np.rint(LI).astype(int)
        labels_list = LI.flatten().tolist()
        c = Counter(labels_list)
        labels_initial = list(c.keys())
        if args.verbose: print('Calling cluster merging:', args.merge_small_clusters_method)
        if args.merge_small_clusters_method == 'none': return LI
        elif args.merge_small_clusters_method == 'k_dependent':
            k_initial = len(labels_initial)
            if -1 in labels_initial: k_initial -= 1
            thresh = k_dependent_smaller_clusters_merging_threshold(k_initial)
        elif args.merge_small_clusters_method == 'fixed':
            thresh = args.fixed_cluster_size_merging_threshold
        else: raise ValueError('Unknown merging method ' + str(args.merge_small_clusters_method))
        assert thresh >= 0.0 and thresh <= 1.0, "Untenable threshold value " + str(thresh)
        if args.verbose: print('Attempting merge. Chose merging threshold', thresh)
        # Now have the threshold, need to merge the smaller clusters below that threshold
        def attempt_merge(currLI):
            """ Runs a single merge iteration. Returns None if nothing was merged. """
            print('\tEntering single merge attempt')
            D = label_img_to_stats(img, img_mask, currLI, (not args.is_mspec))
            cluster_stats = D['cluster_info']['cluster_stats']
            current_labels = D['cluster_info']['allowed_labels']
            found_failure = False
            percents = [ (cluster_stats[clabel]['percent_member_pixels'], clabel)
                         for clabel in current_labels if not clabel == -1 ]
            _current_k = len(percents)
            sorted_percents = sorted(percents, key=lambda x: x[0]) # min percent at front
            if args.verbose: print('\t\tObtained percents', sorted_percents)
            curr_percent, label = sorted_percents[0] # We always merge the smallest one
            # If specified, we should recompute the threshold
            nonlocal thresh
            if args.small_cluster_merging_dynamic_k and args.merge_small_clusters_method == 'k_dependent':
                thresh = k_dependent_smaller_clusters_merging_threshold(_current_k)
                if args.verbose: print('\tRecomputed k-dep threshold:', thresh)
            if curr_percent < thresh:
                if args.verbose: print('\t\tDetected cluster size violation. Merging.')
                found_failure = True
                # Target colour (to destroy)
                targeted_mean_colour = cluster_stats[label]['mean_colour']
                # Distances in colour space to target
                dists = [ (( (  targeted_mean_colour 
                                - cluster_stats[olabel]['mean_colour']
                             )**2 ).sum(), olabel)
                          for olabel in current_labels 
                          if not (olabel in [-1, label]) ]
                sorted_oths = sorted(dists, key=lambda x: x[0], reverse=False) # min dist at front
                # Colour cluster label to replace the target with
                merge_into_target = sorted_oths[0][1] # <- cluster label
                if args.verbose:
                    mean_oth = cluster_stats[merge_into_target]['mean_colour']
                    oth_percent = cluster_stats[merge_into_target]['percent_member_pixels']
                    print('\t\tMerging cluster %d (%s,%.3f) into %d (%s,%.3f)' %
                           (  label, str(targeted_mean_colour), curr_percent,
                              merge_into_target, str(mean_oth), oth_percent   ) )
                # We've found the closest cluster. Now to perform the merging.
                LI_copy = np.copy(currLI)
                LI_copy[ currLI == label ] = merge_into_target
            if found_failure: return LI_copy
            else: return None # Finished all merges
            #</ End of iterated merging routine />#
        # Iteratively attempt merging the smallest cluster until it no longer occurs 
        all_clear = False
        currLI = LI
        while not all_clear:
            merged_output = attempt_merge(currLI)
            if merged_output is None: # we are done
                all_clear = True
            else: # Try again since a merging occurred
                currLI = merged_output
        if args.verbose: print('\tFinished merging attempts')
        return currLI
    # Compute labelling -> merge clusters -> reorder labels
    label_image = reorder_label_img( 
                        merge_smaller_clusters( 
                            label(img, img_mask, args.labeller, args) 
                        ) 
                  )
    if (not args.is_mspec):
        mean_seg_img = color.label2rgb(label_image, image = img, bg_label = -1, 
                                    bg_color = (0.0, 0.0, 0.0), kind = 'avg')
        # Write mean-cluster-valued image out if desired
        if args.write_mean_segs:
            int_mask = (img_mask*255).reshape(H,W,1)
            mean_segs4 = np.concatenate( (mean_seg_img, int_mask), axis = 2 )
            if not os.path.isdir(args.mean_seg_output_dir):
                os.makedirs(args.mean_seg_output_dir)
            cfname = os.path.join(args.mean_seg_output_dir, 
                              img_file_basename_extless + ".mean_seg.png")
            if args.verbose: print('\tSaving mean seg image to', cfname)
            skimage.io.imsave(fname = cfname, arr = mean_segs4)

    ### Compute transition matrix and other stats ###
    M = transition_matrix(label_image, args.normalize_matrix, 
            print_M = (not args.no_print_transitions), args=args )
    img_stats = label_img_to_stats(img, img_mask, label_image, (not args.is_mspec), args.verbose)

    ### Visualization ###
    if args.display: vis_label_img(img, label_image, args)
    plt.show()

    if args.verbose: print('Finished processing', img_path)
    return M, img_stats

def label_img_to_stats(img, mask, label_img, is_color, verbose=False):
    """
    Returns a dictionary of label img and initial image stats
    This includes a subdictionary of information per label set (cluster)
    """
    
    H, W, n_channels = img.shape
    allowed_labels = list(set(label_img.flatten().astype(int).tolist()))
    n_labels = len(allowed_labels)
    n_masked = (label_img == -1).sum()
    n_unmasked = (H*W) - n_masked
    cluster_info = { 'n_labels' : n_labels, 'allowed_labels' : allowed_labels }
    cluster_stats = {}

    for label in allowed_labels:
        if label == -1: continue # Ignore bg
            
        bool_indexer = np.logical_and(mask != 0, label_img == label)
        orig_vals = img[bool_indexer] # Valid values in the current cluster       
        mean_cluster_value = orig_vals.mean(0) # Mean value for each channel of img
       
        # Fill in cluster_stats dictionary
        cluster_stats[label] = {'label_number':label}
        for i in range(len(mean_cluster_value)):
            cluster_stats[label]['mean_value'+str(i+1)] = mean_cluster_value[i]
        if (is_color):
            mean_cluster_value_hsv = rgb2hsv(mean_cluster_value.reshape(1,1,3) / 255.0)[0,0,:]
            cluster_stats[label]['mean_colour_H'] = mean_cluster_value_hsv[0]
            cluster_stats[label]['mean_colour_S'] = mean_cluster_value_hsv[1]
            cluster_stats[label]['mean_colour_V'] = mean_cluster_value_hsv[2]
        cluster_stats[label]['n_member_pixels'] = orig_vals.shape[0]
        cluster_stats[label]['percent_member_pixels'] = orig_vals.shape[0] / n_unmasked
        cluster_stats[label]['mean_colour'] = mean_cluster_value
            
    cluster_info['cluster_stats'] = cluster_stats
    D = { 'H'                 : H,
          'W'                 : W,
          'total_pixels'      : H*W,
          'n_masked_pixels'   : n_masked,
          'n_unmasked_pixels' : n_unmasked,
          'cluster_info'      : cluster_info,
    }
    if verbose: 
        print('Gathered seg image info:')
        _glob_form.print_dict(D)
    return D

def label(image, mask, method, args):
    """
    Input: I (M x N x C)
    Output: Pixelwise labels (M x N), integer
    """
    # Convert to desired colour space
    if (not args.is_mspec):
        if not args.clustering_colour_space == 'rgb':
            if args.verbose: 
                _K = args.clustering_colour_space 
                _app = { 'cie' : '(CIE-1931-RGB)', 'lab' : '(CIE-LAB)', 'hsv' : '' }
                print('Converting to colour space', _K, _app[_K]) 
            if args.clustering_colour_space == 'hsv':
                image = np.rint( rgb2hsv(image/255.0) * 255.0 ).astype(int) 
            if args.clustering_colour_space == 'cie': # CIE1931
                from skimage.color import rgb2rgbcie
                image = np.rint( rgb2rgbcie(image/255.0) * 255.0 ).astype(int)
            if args.clustering_colour_space == 'lab': # CIE-LAB / CIE L*a*b*
                # LAB may be the most perceptually principled choice since (in humans)
                # it can be used to compute more perceptual colour differences.
                from skimage.color import rgb2lab # Bounds: [0,100], [-128,128], [-128,128]
                image = np.rint( rgb2lab(image/255.0) ).astype(int) 
    # Perform clustering
    if method == 'graph_cuts': return segment(image, method, mask, args)
    else:                      return cluster_based_img_seg(image, mask, method, args)

def cluster_vecs(X, method, args):
    """
    Input: X (S x D, row vector per datum)
    Output: cluster labels vector (S, integer)
    """
    X = X.astype(float)
    if   method == 'affinity_prop': 
        clusterer = cluster.AffinityPropagation(affinity = 'euclidean',
                                                damping = 0.5) 
    elif method == 'dbscan':          
        clusterer = cluster.DBSCAN(eps = args.dbscan_eps, 
                                   min_samples = args.dbscan_min_neb_size,
                                   metric = 'euclidean') 
    elif method == 'optics':          
        clusterer = cluster.OPTICS(min_samples = 0.05,
                                   max_eps = np.inf,
                                   xi = 0.05)
    elif method == 'kmeans':
        kmeans_B = 100
        specified_k = args.kmeans_k
        if not (specified_k is None):
            if args.verbose: print("\tPreparing k-means clusterer with fixed k =", specified_k)
            clusterer = cluster.MiniBatchKMeans(n_clusters = int(args.kmeans_k), batch_size = kmeans_B)
        else:
            _min_k, _max_k = list(map(int, args.kmeans_auto_bounds.strip().split(","))) # min,max
            verb_str_a = "\tAttempting to automatically determine k. Searching over k ="
            if args.verbose: print(verb_str_a, _min_k, "to", _max_k)
            def run_clustering(nc_k):
                if args.verbose: print("\tOn k =", nc_k)
                return cluster.MiniBatchKMeans(n_clusters = nc_k, batch_size = kmeans_B)
            clusterers = [ (i, run_clustering(i)) 
                            for i in range(_min_k, _max_k+1) ]
            clustering_labels = [ (c[0], c[1].fit_predict(X)) for c in clusterers ]
            if args.kmeans_auto_crit == 'silhouette': 
                scorer_function = sklearn.metrics.silhouette_score
                _coef = 1.0
            elif args.kmeans_auto_crit == 'davies_bouldin': 
                scorer_function = sklearn.metrics.davies_bouldin_score
                _coef = -1.0 # Since lower is better
            elif args.kmeans_auto_crit == 'calinski_harabasz':
                scorer_function = sklearn.metrics.calinski_harabasz_score
                _coef = 1.0
            else:
                raise ValueError('Unknown scorer ' + args.kmeans_auto_crit)
            crit_scores = [ (i, scorer_function(X, labels)) for (i, labels) in clustering_labels ]
            best_ind = np.argmax([ _coef * a[1] for a in crit_scores ])
            if args.verbose:
                print("\tObtained the following scores with criterion %s:" % args.kmeans_auto_crit)
                for i, s in crit_scores:
                    print("\t\tk = %d -> %.3f" % (i,s))
                print('\tBest k:', crit_scores[best_ind][0])
            return clustering_labels[best_ind][1]
    else:
        raise ValueError('Unknown method' + method)
    if args.verbose: print("\tRunning clustering via", method)
    return clusterer.fit_predict(X)

def cluster_based_img_seg(image, mask, method, args):
    """
    Use spatially unaware vector-space clustering for label assignment.
    BG label: -1
    """
    # Extract colour-based vectors
    # TODO micropatches option
    vecs        = extract_vecs(image, mask) 
    if args.verbose: print("\tObtained vectors:", vecs.shape)
    # Cluster colour vectors to obtain labels
    vec_labels  = cluster_vecs(vecs, method, args) 
    if args.verbose: print("\t\tComplete")
    # Use the image positions to reform the image
    # Background pixels are labeled -1
    label_image = reform_label_image(vec_labels, mask) 
    if args.verbose: print("\tFinished reforming label image")
    return label_image

def extract_vecs(image, mask): return image[ mask > 0 ]

def reform_label_image(vec_labels, mask):
    H, W = mask.shape
    # Form a label image (initialize to -2)
    label_image = np.zeros( (H,W) ) - 2
    # Set unmasked parts
    label_image[mask > 0] = vec_labels
    ### Post-process to remove noise labels ###
    # E.g., noise labels are -1 in OPTICS and DBSCAN -> convert to nan
    label_image[ label_image == -1 ] = np.nan
    # Convert to masked array
    a = np.ma.masked_invalid(label_image)
    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
    # Retrieve only valid values
    x1 = xx[ ~a.mask ] # x indices
    y1 = yy[ ~a.mask ] # y indices
    new_data = a[ ~a.mask ] # unmasked data values
    # Interpolate via nearest neighbours
    G = interpolate.griddata( (x1, y1), new_data.ravel(),
                              (xx, yy), method = 'nearest' )
    label_image = G
    ### Move -2 -> -1 (background signal) ###
    label_image[ label_image == -2 ] = -1
    assert not np.isnan(label_image).any() 
    return label_image

def vis_label_img(image, labels_img, args):
    """
    Visualization method for clustered/segmented image.
    Display segmentation, overlaid segmentation, and mean-value segmentation.
    """
    if (args.is_mspec):
        # Segmentation
        pure_segs = color.label2rgb(labels_img)
        # Overlaid segmentation
        over_segs = color.label2rgb(labels_img, image=image, 
                                alpha=0.25, bg_label=-1, 
                                bg_color=(0,0,0), image_alpha=1, 
                                kind='overlay')
        # Mean value segmentation
        mean_segs = color.label2rgb(labels_img, image=image, 
                                alpha=0.25, bg_label=-1, 
                                bg_color=(0,0,0), image_alpha=1, 
                                kind='avg')
        # Show image triplet
        fig, ax = plt.subplots(nrows=2, ncols=2) #, sharex=True, sharey=True) #, figsize=(25, 8))
        _q = [(0,0), (0,1), (1,0), (1,1)]
        _t = ['Orig Image', 'Segmentation', 'Seg Overlay', 'Mean Seg']
        for i, I in enumerate([image, pure_segs, over_segs, mean_segs]):
            j, k = _q[i]
            ax[j, k].imshow(I)
            ax[j, k].axis('off')
            ax[j, k].set_title(_t[i])
        plt.tight_layout()
    else:
        # Segmentation
        pure_segs = color.label2rgb(labels_img)
        plt.imshow(pure_segs)
        plt.title('Segmentation')
        plt.axis('off')
        plt.tight_layout()

def transition_matrix(L, normalize, print_M, args):
    """
    Input: label image (M x N, integer in [1,K])
    Output: K x K transition count/probability matrix (M_ij = count(i->j))
    """
    H, W = L.shape
    Hpp, Wpp = H + 1, W + 1
    allowed_labels = list(set(L.flatten().astype(int).tolist()))
    n_labels = len(allowed_labels)
    n_masked = (L == -1).sum()
    # Shifted label images
    L_left  = np.c_[ L, -np.ones(H) ]
    L_right = np.c_[ -np.ones(H), L ]
    L_up    = np.r_[ [-np.ones(W)], L ]
    L_down  = np.r_[ L, [-np.ones(W)] ]
    horz_shift = np.stack( (L_left, L_right), axis=-1 ).reshape(H*Wpp, 2)
    vert_shift = np.stack( (L_up, L_down),    axis=-1 ).reshape(Hpp*W, 2)
    all_tuples = np.concatenate( (horz_shift, vert_shift), axis = 0 ).astype(int).tolist()
    if args.verbose: print('\tAssembled', len(all_tuples), 'edge pairs for transition estimation')
    if not args.keep_bg:
        if -1 in allowed_labels: allowed_labels.remove(-1) # 
        n_labels = len(allowed_labels)
        _ATN = len(all_tuples)
        all_tuples = [ elem for elem in all_tuples if not (-1 in elem) ]
        if args.verbose: 
            print("\t\tPost-bg removal:", len(all_tuples), "remaining (%d removed)" % 
                    (_ATN - len(all_tuples)))
            print("\t\tExpected", 2*(H+W), 'missing values due to appended boundaries')
    # Stringify the tuples
    all_tuples = map(lambda s: ",".join(map(str, s)), all_tuples)
    transition_counters = Counter(all_tuples)
    # Generate matrix form of transition
    M = np.zeros( (n_labels, n_labels) )
    for i, labi in enumerate(allowed_labels):
        for j, labj in enumerate(allowed_labels):
            key1 = str(labi) + "," + str(labj)
            key2 = str(labj) + "," + str(labi)
            M[i,j] = int(transition_counters[key1] + transition_counters[key2])
    if normalize: M = M / M.sum(axis=1)[:,np.newaxis]
    else:         M = M.astype(int)
    if print_M or args.verbose: 
        if args.verbose: print("Transition counts:", transition_counters)
        print('Labels:', allowed_labels)
        if args.verbose: 
            print('Column sums:', M.sum(1), '| Total:', np.tril(M,k=0).sum())
            extot = 2*H*W + W + H
            print('N_expected_transitions:', extot, '(unmasked: %d)' % (extot - n_masked - H - W))
        print('Transition matrix')
        print(M)
    return M, transition_counters

def segment(image, method, mask, args):
    """ Spatially aware image segmentation """
    #if not (mask is None):
    #    H, W, C = image.shape
    #    image = np.concatenate( (image, mask.reshape(H,W,1)), axis=2)
    image = send_mask_to_minus_one(image, mask)
    if method == 'graph_cuts':
        labels1 = segmentation.slic(image, compactness=args.gc_compactness,
                                    n_segments=args.gc_n_segments, 
                                    sigma=args.gc_slic_sigma) #, compactness=gc_compactness)
        out1    = color.label2rgb(labels1, image, kind='avg')
        g       = graph.rag_mean_color(image, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
        labels2 = send_mask_to_minus_one(labels2, mask)
        return labels2

def gauss_filter(img, blur_sigma):
    assert blur_sigma >= 0.0, "Untenable blur kernel width"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_ubyte( utils.gaussian_blur(img, blur_sigma) )
    return img

def send_mask_to_minus_one(image, mask):
    image[ mask <= 0 ] = -1 
    return image

#-------------------------#
if __name__ == '__main__':
    main()
#-------------------------#


#
