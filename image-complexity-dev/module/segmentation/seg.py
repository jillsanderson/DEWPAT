import argparse, sklearn, skimage, numpy as np, numpy.random as npr
import warnings, matplotlib.pyplot as plt, csv
from skimage import data, segmentation, color
from skimage.color import rgb2hsv, rgb2lab
from skimage.future import graph
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from scipy import interpolate
from collections import Counter
from ..utils import Formatter, gaussian_blur


### Label (cluster/segment) the image ###
def reorder_label_img(LI, verbose=False):
    """ Reorders the labels in 'LI'.

    Given N clusters, the clusters are ordered by decreasing size and
    assigned labels from 1 to N. The background label -1 is not changed.

    Args:
        LI: A 2D array of labels.
        verbose: Optional; Print verbosely if True. False by default.
    
    Returns:
        A 2D array of the reordered labels.
    """
    if verbose: print('\tReordering label image')
    LI = np.rint(LI).astype(int)
    labels_list = LI.flatten().astype(int).tolist()
    c = Counter(labels_list)
    if verbose: print('\tOrig Counter:', c)
    ordered_targets = c.most_common() # [(int_key, count),...]
    curr_new_label = 1
    LI_new = np.copy(LI)
    for j, (int_key, count) in enumerate(ordered_targets):
        if int_key == -1: continue # Leave the background alone
        LI_new[ LI == int_key ] = curr_new_label
        curr_new_label += 1
    if verbose: print('\tNew Counter:', Counter(LI_new.flatten().astype(int).tolist()))
    return LI_new

def k_dependent_smaller_clusters_merging_threshold(k, kdep_param=0.05):
    """ Returns the k-dependent cluster merging threshold.

    The threshold is given by
        T0 / (k-1)
    where T0 is the initial threshold value for k=2 clusters and k
    is the number of clusters.

    Args:
        k: The number of clusters.
        kdep_param: The initial threshold value for k=2 clusters.
          0.05 by default.
    """
    # Start at some value for k=2, and anneal down as k increases
    if k <= 1: return 0.0 # No merging if k == 1
    T0 = kdep_param
    T  = T0 / (k - 1)
    return T

def merge_smaller_clusters(img, mask, LI, merge_small_clusters_method=None, 
                           small_cluster_merging_kdep_param=0.05,
                           fixed_cluster_size_merging_threshold=0.05, 
                           small_cluster_merging_dynamic_k=False, verbose=False):
    """ Merges clusters with sizes below a specified threshold.

    The function iteratively attempts to merge the smallest cluster until it is no longer
    below the threshold. The new labels for the cluster are determined by the label of the
    cluster with the nearest (euclidean) mean pixel value.

    The background label -1 is never merged.

    'merge_small_clusters_method' must be one of the following:
        "fixed"       - Threshold is the same for each iteration.
        "k-dependent" - Threshold depends on the number of clusters.
        None          - No merging is performed (original labels are returned).

    All thresholds below represent a percentage and must be in [0.0, 1.0].

    Args:
        img: The input image.
        mask: A binary alpha mask.
        LI: A 2D array of labels (clusters) for 'img'.
        merge_small_clusters_method: Optional; The method used to merge the clusters.
          Must be "fixed", "k-dependent", or None (default).
        small_cluster_merging_kdep_param: Optional; If 'merge_small_clusters_method' is
          "k-dependent", this is the initial threshold for k=2 clusters. 0.05 by default.
        fixed_cluster_size_merging_threshold: Optional; If 'merge_small_clusters_method' is
          "fixed", this is the fixed threshold used for each iteration. 0.05 by default.
        small_cluster_merging_dynamic_k: Optional; If True and 'merge_small_clusters_method'
          is "k-dependent", the threshold will be dynamically updated as clusters are removed
          each iteration. False by default.
        verbose: Optional; Print verbosely if True. False by default.

    Returns:
        A 2D array of the updated labels.

    Raises:
        ValueError: 'merge_small_clusters_method' is not recognized.
    """
    LI = np.rint(LI).astype(int)
    labels_list = LI.flatten().tolist()
    c = Counter(labels_list)
    labels_initial = list(c.keys())
    if verbose: print('Calling cluster merging:', merge_small_clusters_method)
    if merge_small_clusters_method == 'none': return LI
    elif merge_small_clusters_method == 'k_dependent':
        k_initial = len(labels_initial)
        if -1 in labels_initial: k_initial -= 1
        thresh = k_dependent_smaller_clusters_merging_threshold(k_initial, 
                                kdep_param=small_cluster_merging_kdep_param)
    elif merge_small_clusters_method == 'fixed':
        thresh = fixed_cluster_size_merging_threshold
    else: raise ValueError('Unknown merging method ' + str(merge_small_clusters_method))
    assert thresh >= 0.0 and thresh <= 1.0, "Untenable threshold value " + str(thresh)
    if verbose: print('Attempting merge. Chose merging threshold', thresh)
    # Now have the threshold, need to merge the smaller clusters below that threshold
    def attempt_merge(currLI):
        """ Runs a single merge iteration. Returns None if nothing was merged. """
        print('\tEntering single merge attempt')
        D = label_img_to_stats(img, mask, currLI)
        cluster_stats = D['cluster_info']['cluster_stats']
        current_labels = D['cluster_info']['allowed_labels']
        found_failure = False
        percents = [ (cluster_stats[clabel]['percent_member_pixels'], clabel)
                        for clabel in current_labels if not clabel == -1 ]
        _current_k = len(percents)
        sorted_percents = sorted(percents, key=lambda x: x[0]) # min percent at front
        if verbose: print('\t\tObtained percents', sorted_percents)
        curr_percent, label = sorted_percents[0] # We always merge the smallest one
        # If specified, we should recompute the threshold
        nonlocal thresh
        if small_cluster_merging_dynamic_k and merge_small_clusters_method == 'k_dependent':
            thresh = k_dependent_smaller_clusters_merging_threshold(_current_k)
            if verbose: print('\tRecomputed k-dep threshold:', thresh)
        if curr_percent < thresh:
            if verbose: print('\t\tDetected cluster size violation. Merging.')
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
            if verbose:
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
    if verbose: print('\tFinished merging attempts')
    return currLI

def label_img_to_stats(img, mask, label_img, mean_seg_img=None, medians=None,
                       modes=None, gmm_means=None, gmm_weights=None, verbose=False):
    """ Returns a dictionary of 'label_img' and initial image stats.

    Includes a subdictionary of information per label set (cluster).

    Args:
        img: The input image.
        mask: The alpha mask of the image.
        label_img: A 2D array of labels for the image.
        mean_seg_img: Optional; 'label_img' replaced with the mean cluster value
          for each label. If set to None (default), the image will be generated
          in the function.
        verbose: Optional; Print verbosely if True. False by default.
    
    Returns:
        A dictionary of the image stats.
    """
    if mean_seg_img is None:
        mean_seg_img = color.label2rgb(label_img, image = img, bg_label = -1, 
                                       bg_color = (0.0, 0.0, 0.0), kind = 'avg')
    if medians is None:
        _, medians = median_label2rgb(img, label_img)
    if modes is None:
        _, modes, gmm_means, gmm_weights = gmm_label2rgb(img, label_img)
    # 
    H, W, C        = img.shape
    allowed_labels = list(set(label_img.flatten().astype(int).tolist()))
    n_labels       = len(allowed_labels)
    n_masked       = (label_img == -1).sum()
    n_unmasked     = (H*W) - n_masked
    cluster_info   = { 'n_labels' : n_labels, 'allowed_labels' : allowed_labels }
    cluster_stats  = {}
    median_cluster_stats = {}
    mode_cluster_stats   = {}
    for label in allowed_labels:
        if label == -1: continue # Ignore bg
        bool_indexer = np.logical_and(mask != 0, label_img == label)
        orig_vals = img[bool_indexer] # Valid values in the current cluster
        mean_vals = mean_seg_img[bool_indexer]
        mean_cluster_value = orig_vals.mean(0)
        premeaned_val = mean_vals.mean(0)
        premeaned_val_hsv = rgb2hsv(premeaned_val.reshape(1,1,3) / 255.0)[0,0,:]
        premeaned_val_lab = rgb2lab(premeaned_val.reshape(1,1,3) / 255.0)[0,0,:]
        cluster_stats[label] = {
            'label_number'          : label,
            'mean_C1'               : mean_cluster_value[0], # R (floats)
            'mean_C2'               : mean_cluster_value[1], # G
            'mean_C3'               : mean_cluster_value[2], # B
            'mean_C1_hsv'           : premeaned_val_hsv[0],  # H in [0,1], multiply by 360 to get degrees
            'mean_C2_hsv'           : premeaned_val_hsv[1],  # S
            'mean_C3_hsv'           : premeaned_val_hsv[2],  # V
            'mean_C1_lab'           : premeaned_val_lab[0],  # L (floats)
            'mean_C2_lab'           : premeaned_val_lab[1],  # A
            'mean_C3_lab'           : premeaned_val_lab[2],  # B
            'n_member_pixels'       : orig_vals.shape[0],
            'percent_member_pixels' : orig_vals.shape[0] / n_unmasked,
            'mean_colour'           : premeaned_val, # Integer array
        }

        median_cluster_value = medians[label-1]
        if verbose: print(label, 'Median cluster value', median_cluster_value)
        median_hsv = rgb2hsv(median_cluster_value.reshape(1,1,3) / 255.0)[0,0,:]
        median_lab = rgb2lab(median_cluster_value.reshape(1,1,3) / 255.0)[0,0,:]
        median_cluster_stats[label] = {
            'label_number'            : label,
            'median_C1'               : median_cluster_value[0], # R (ints)
            'median_C2'               : median_cluster_value[1], # G
            'median_C3'               : median_cluster_value[2], # B
            'median_C1_hsv'           : median_hsv[0],  # H in [0,1], multiply by 360 to get degrees
            'median_C2_hsv'           : median_hsv[1],  # S
            'median_C3_hsv'           : median_hsv[2],  # V
            'median_C1_lab'           : median_lab[0],  # L (floats)
            'median_C2_lab'           : median_lab[1],  # A
            'median_C3_lab'           : median_lab[2],  # B
            'n_member_pixels'       : orig_vals.shape[0],
            'percent_member_pixels' : orig_vals.shape[0] / n_unmasked
        }

        mode_cluster_stats[label] = {
            'label_number'          : label,
            'n_member_pixels'       : orig_vals.shape[0],
            'percent_member_pixels' : orig_vals.shape[0] / n_unmasked
        }

        cluster_modes = modes[label-1]
        cluster_means = gmm_means[label-1]
        cluster_weights = gmm_weights[label-1]
        # get RGB, HSV, LAB for each mode
        for i in range(1, len(cluster_modes)+1):
            mode = cluster_modes[i-1]
            mean = cluster_means[i-1]
            weight = cluster_weights[i-1]
            if verbose: print(label, f'Cluster mode {i} value', mode)
            mode_hsv = rgb2hsv(mode.reshape(1,1,3) / 255.0)[0,0,:]
            mode_lab = rgb2lab(mode.reshape(1,1,3) / 255.0)[0,0,:]

            _mode_stats = {
                f'mode{i}_C1'       :  mode[0],     # R (ints)
                f'mode{i}_C2'       :  mode[1],     # G
                f'mode{i}_C3'       :  mode[2],     # B
                f'mode{i}_C1_hsv'   :  mode_hsv[0], # H in [0,1], multiply by 360 to get degrees
                f'mode{i}_C2_hsv'   :  mode_hsv[1], # S
                f'mode{i}_C3_hsv'   :  mode_hsv[2], # V
                f'mode{i}_C1_lab'   :  mode_lab[0], # L (floats)
                f'mode{i}_C2_lab'   :  mode_lab[1], # A
                f'mode{i}_C3_lab'   :  mode_lab[2], # B
                f'gmm{i}_C1_mean'   :  mean[0],     # GMM component mean R (float)
                f'gmm{i}_C2_mean'   :  mean[1],     # G
                f'gmm{i}_C3_mean'   :  mean[2],     # B
                f'gmm{i}_weight'    :  weight       # GMM component weight (float)
            }

            mode_cluster_stats[label] = {**mode_cluster_stats[label], **_mode_stats}
    
    cluster_info['cluster_stats'] = cluster_stats
    cluster_info['median_cluster_stats'] = median_cluster_stats
    cluster_info['mode_cluster_stats'] = mode_cluster_stats
    D = { 'H'                 : H,
          'W'                 : W,
          'total_pixels'      : H*W,
          'n_masked_pixels'   : n_masked,
          'n_unmasked_pixels' : n_unmasked,
          'cluster_info'      : cluster_info,
    }
    if verbose: 
        print('Gathered seg image info:')
        _glob_form = Formatter()
        _glob_form.print_dict(D)
    return D

def label(image, mask, method, clustering_colour_space='rgb', dbscan_eps=1.0, 
          dbscan_min_neb_size=20, kmeans_k=None,
          kmeans_auto_bounds='2,6', kmeans_auto_crit='davies_bouldin', 
          gc_compactness=10.0, gc_n_segments=300,
          gc_slic_sigma=0.0,verbose=False):
    """ Segments the image using the specified method.

    The possible methods are:

    graph_cuts    - Spatially unaware segmentation; K-means clustering
                    followed by a normalized graph cut.
    affinity_prop - Affinity propagation clustering. Similarity between
                    two pixels is defined as the negative squared 
                    euclidean distance.
    dbscan        - DBSCAN using euclidean distance.
    optics        - OPTICS using minkowski distance.
    kmeans        - Mini-batch k-means clustering using batch sizes of 100.
                    If k is not given, k will be automatically
                    determined based on a specified scoring function
                    (see 'kmeans_auto_crit').

    Args:
        image: The input image.
        mask: A binary alpha mask.
        method: The method of segmentation. Must be one of "graph_cuts",
          "affinity_prop", "dbscan", "optics", or "kmeans".
        clustering_colour_space: Optional; The colour space in which the
          clustering is performed. Can be "rgb" (default), "hsv", "cie",
          or "lab".
        dbscan_eps: Optional; If the method is "dbscan", this is the
          maximum distance between two points for one to be considered as
          in the neighborhood of the other. 1.0 by default.
        dbscan_min_neb_size: Optional; If the method is "dbscan", this is the
          number of samples (or total weight) in a neighborhood for a point to
          be considered as a core point. 20 by default.
        kmeans_k: Optional; If method is "kmeans", this is the number of clusters.
          If kmeans_k is None, k will be automatically determined. None by default.
        kmeans_auto_bounds: Optional; If method is "kmeans", this is a string in
          the form "X,Y" which represents the search range of k values, [X, Y].
          "2,6" by default.
        kmeans_auto_crit: Optional; If method is "kmeans" and k is not fixed,
          this is the metric used to score the results of the different k values.
          Can be "silhouette", "davies_bouldin" (default), or "calinski_harabasz".
        gc_compactness: Optional; If method is "graph_cuts", this is the weight given
          to space proximity (as opposed to colour proximity). Higher values result in
          higher superpixel compactness. 10.0 by default.
        gc_n_segments: Optional; If method is "graph_cuts", this is the approximate
          number of clusters created during the k-means clustering. 300 by default.
        gc_slic_sigma: Optional; If method is "graph_cuts", this is the width of the
          Gaussian smoothing kernel applied to each colour channel. If gc_slic_sigma
          is 0 (default), no pre-processing is done.
        verbose: Optional; Print verbosely if True. False by default.

    Returns:
        A 2D array of integer labels corresponding to the original image. Background pixels
        are given a label of -1.
    """
    # Convert to desired colour space
    if not clustering_colour_space == 'rgb':
        if verbose: 
            _K = clustering_colour_space 
            _app = { 'cie' : '(CIE-1931-RGB)', 'lab' : '(CIE-LAB)', 'hsv' : '' }
            print('Converting to colour space', _K, _app[_K]) 
        if clustering_colour_space == 'hsv':
            image = np.rint( rgb2hsv(image/255.0) * 255.0 ).astype(int) 
        if clustering_colour_space == 'cie': # CIE1931
            from skimage.color import rgb2rgbcie
            image = np.rint( rgb2rgbcie(image/255.0) * 255.0 ).astype(int)
        if clustering_colour_space == 'lab': # CIE-LAB / CIE L*a*b*
            # LAB may be the most perceptually principled choice since (in humans)
            # it can be used to compute more perceptual colour differences.
            from skimage.color import rgb2lab # Bounds: [0,100], [-128,128], [-128,128]
            image = np.rint( rgb2lab(image/255.0) ).astype(int) 
    # Perform clustering
    if method == 'graph_cuts': return segment(image, method, mask, 
                                              gc_compactness=gc_compactness, 
                                              gc_n_segments=gc_n_segments,
                                              gc_slic_sigma=gc_slic_sigma)
    else:                      return cluster_based_img_seg(image, mask, method, dbscan_eps=dbscan_eps,
                                                            dbscan_min_neb_size=dbscan_min_neb_size, 
                                                            kmeans_k=kmeans_k,
                                                            kmeans_auto_bounds=kmeans_auto_bounds,
                                                            kmeans_auto_crit=kmeans_auto_crit, 
                                                            verbose=verbose)

def cluster_vecs(X, method, dbscan_eps=1.0, dbscan_min_neb_size=20, kmeans_k=None, kmeans_auto_bounds='2,6',
                 verbose=False, kmeans_auto_crit='davies_bouldin'):
    """ Clusters a vector of pixels using the specified method.

    The possible methods are:

    affinity_prop - Affinity propagation clustering. Similarity between
                    two pixels is defined as the negative squared 
                    euclidean distance.
    dbscan        - DBSCAN using euclidean distance.
    optics        - OPTICS using minkowski distance.
    kmeans        - Mini-batch k-means clustering using batch sizes of 100.
                    If k is not given, k will be automatically
                    determined based on a specified scoring function
                    (see 'kmeans_auto_crit').

    Args:
        X: S X D, row vector per datum.
        method: A string representing the method. Can be "affinity_prop",
          "dbscan", "optics", or "kmeans".
        dbscan_eps: Optional; If the method is "dbscan", this is the
          maximum distance between two points for one to be considered as
          in the neighborhood of the other. 1.0 by default.
        dbscan_min_neb_size: Optional; If the method is "dbscan", this is the
          number of samples (or total weight) in a neighborhood for a point to
          be considered as a core point. 20 by default.
        kmeans_k: Optional; If method is "kmeans", this is the number of clusters.
          If kmeans_k is None (default), k will be automatically determined.
        kmeans_auto_bounds: Optional; If method is "kmeans", this is a string in
          the form "X,Y" which represents the search range of k values, [X, Y].
          "2,6" by default.
        kmeans_auto_crit: Optional; If method is "kmeans" and k is not fixed,
          this is the metric used to score the results of the different k values.
          Can be one of "silhouette", "davies_bouldin" (default), or "calinski_harabasz".
        verbose: Optional; Print verbosely if True. False by default.
    
    Returns:
        A vector of labels.
    
    Raises:
        ValueError: method or kmeans_auto_crit is not recognized.
    """
    X = X.astype(float)
    if   method == 'affinity_prop': 
        clusterer = cluster.AffinityPropagation(affinity = 'euclidean',
                                                damping = 0.5) 
    elif method == 'dbscan':          
        clusterer = cluster.DBSCAN(eps = dbscan_eps, 
                                   min_samples = dbscan_min_neb_size,
                                   metric = 'euclidean') 
    elif method == 'optics':          
        clusterer = cluster.OPTICS(min_samples = 0.05,
                                   max_eps = np.inf,
                                   xi = 0.05)
    elif method == 'kmeans':
        kmeans_B = 100
        specified_k = kmeans_k
        if not (specified_k is None):
            if verbose: print("\tPreparing k-means clusterer with fixed k =", specified_k)
            clusterer = cluster.MiniBatchKMeans(n_clusters = int(kmeans_k), batch_size = kmeans_B)
        else:
            _min_k, _max_k = list(map(int, kmeans_auto_bounds.strip().split(","))) # min,max
            verb_str_a = "\tAttempting to automatically determine k. Searching over k ="
            if verbose: print(verb_str_a, _min_k, "to", _max_k)
            def run_clustering(nc_k):
                if verbose: print("\tOn k =", nc_k)
                return cluster.MiniBatchKMeans(n_clusters = nc_k, batch_size = kmeans_B)
            clusterers = [ (i, run_clustering(i)) 
                            for i in range(_min_k, _max_k+1) ]
            clustering_labels = [ (c[0], c[1].fit_predict(X)) for c in clusterers ]
            if kmeans_auto_crit == 'silhouette': 
                scorer_function = sklearn.metrics.silhouette_score
                _coef = 1.0
            elif kmeans_auto_crit == 'davies_bouldin': 
                scorer_function = sklearn.metrics.davies_bouldin_score
                _coef = -1.0 # Since lower is better
            elif kmeans_auto_crit == 'calinski_harabasz':
                scorer_function = sklearn.metrics.calinski_harabasz_score
                _coef = 1.0
            else:
                raise ValueError('Unknown scorer ' + kmeans_auto_crit)
            crit_scores = [ (i, scorer_function(X, labels)) for (i, labels) in clustering_labels ]
            best_ind = np.argmax([ _coef * a[1] for a in crit_scores ])
            if verbose:
                print("\tObtained the following scores with criterion %s:" % kmeans_auto_crit)
                for i, s in crit_scores:
                    print("\t\tk = %d -> %.3f" % (i,s))
                print('\tBest k:', crit_scores[best_ind][0])
            return clustering_labels[best_ind][1]
    else:
        raise ValueError('Unknown method' + method)
    if verbose: print("\tRunning clustering via", method)
    return clusterer.fit_predict(X)

def cluster_based_img_seg(image, mask, method, dbscan_eps=1.0, dbscan_min_neb_size=20, kmeans_k=None,
                          kmeans_auto_bounds='2,6', kmeans_auto_crit='davies_bouldin', verbose=False):
    """ Performs spatially unaware vector-space clustering for label assignment.
    
    See cluster_vecs() for more details.

    Args:
        image: The input image.
        mask: A binary alpha mask.
        method: A string representing the method. Can be "affinity_prop",
          "dbscan", "optics", or "kmeans".
        dbscan_eps: Optional; If the method is "dbscan", this is the
          maximum distance between two points for one to be considered as
          in the neighborhood of the other. 1.0 by default.
        dbscan_min_neb_size: Optional; If the method is "dbscan", this is the
          number of samples (or total weight) in a neighborhood for a point to
          be considered as a core point. 20 by default.
        kmeans_k: Optional; If method is "kmeans", this is the number of clusters.
          If kmeans_k is None (default), k will be automatically determined.
        kmeans_auto_bounds: Optional; If method is "kmeans", this is a string in
          the form "X,Y" which represents the search range of k values, [X, Y].
          "2,6" by default.
        kmeans_auto_crit: Optional; If method is "kmeans" and k is not fixed,
          this is the metric used to score the results of the different k values.
          Can be one of "silhouette", "davies_bouldin" (default), or "calinski_harabasz".
        verbose: Optional; If True, print verbosely. False by default.
    
    Returns:
        A 2D array of integer labels corresponding to the original image. Background pixels
        are given a label of -1.
    
    Raises:
        ValueError: method or kmeans_auto_crit is not recognized.
    """
    # Extract colour-based vectors
    # TODO micropatches option
    vecs        = extract_vecs(image, mask) 
    if verbose: print("\tObtained vectors:", vecs.shape)
    # Cluster colour vectors to obtain labels
    vec_labels  = cluster_vecs(vecs, method, dbscan_eps=dbscan_eps, dbscan_min_neb_size=dbscan_min_neb_size, kmeans_k=kmeans_k,
                               kmeans_auto_bounds=kmeans_auto_bounds, kmeans_auto_crit=kmeans_auto_crit, verbose=verbose) 
    if verbose: print("\t\tComplete")
    # Use the image positions to reform the image
    # Background pixels are labeled -1
    label_image = reform_label_image(vec_labels, mask) 
    if verbose: print("\tFinished reforming label image")
    return label_image

def extract_vecs(image, mask): return image[ mask > 0 ]

def reform_label_image(vec_labels, mask):
    """ Reforms the image using the image position of the labels.

    Args:
        vec_labels: A vector of labels.
        mask: A binary alpha mask.

    Returns:
        A 2D array of labels.
    """
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


def _cluster_median(cluster):
    """ Returns the pixel with the minimum L1 distance sum to all other pixels in the cluster.

    Args:
        cluster: A 1D array of pixels.
    """
    # efficient O(n log n) implementation
    # https://stackoverflow.com/a/12905913
    total_dist = np.zeros(len(cluster))
    C = cluster[0].shape[0]

    # TODO: consider vectorizing
    for channel in range(C):
        values = cluster[:,channel]
        ind = np.argsort(values)

        n = len(ind)
        dist = np.zeros(n)

        # The distance between any index j < i and i can be expressed as the distance from j to
        # i-1 plus the distance from i-1 to i. Let D(i) denote the total distance between all
        # j < i and i (the total distance to every point left of i). Using the fact above,
        # D(i) can be computed by taking D(i-1) and adding values[i] - values[i-1] (the distance
        # from i-1 to i) for each index 0,1,...,i-1.
        _sum = 0
        for i in range(1, n):
            _sum += (values[ind[i]] - values[ind[i-1]])*i
            # _sum is now the total distance from point i to all j < i
            dist[ind[i]] += _sum

        # Repeat process above but for distance between i and every point to the right of it.
        _sum = 0
        for i in range(n-2, -1, -1):
            _sum += (values[ind[i+1]] - values[ind[i]])*(n-i-1)
            # _sum is now the total distance from point i to all j > i
            dist[ind[i]] += _sum

        total_dist += dist

    return cluster[np.argmin(total_dist)]


def median_label2rgb(image, labels_img, bg_color=(0,0,0)):
    """ Paint the clusters with its median pixel value.

    Args:
        image: The image object.
        labels_img: The labels given to 'image' after segmentation.
        bg_color: Optional; the RGB color given to the background labels.
    
    Returns:
        The painted image and a list of median pixel values per cluster.
    """
    allowed_labels = list(set(labels_img.flatten().astype(int).tolist()))

    cluster_medians = []
    H, W = image.shape[:2]
    for label in allowed_labels:
        if label == -1: continue
        cluster = np.array([image[i,j] for i in range(H) for j in range(W) if labels_img[i,j] == label])
        cluster_medians.append(_cluster_median(cluster))
    
    return _paint_img(image, labels_img, cluster_medians, bg_color=bg_color), cluster_medians


def _cluster_gmm(cluster, n_components=3):
    """ Fits pixels in 'cluster' to GMM and returns the set of modes for the cluster.

    The modes of the cluster are determined by replacing the means of the GMM
    components with the nearest pixel in the cluster (L1 distance).

    Args:
        cluster: A 1D array of pixels.
        n_components: Optional; the number of components in the GMM.
    
    Returns:
        A tuple containing
            1) A list of the mode pixels.
            2) A list of the components' mean pixel values.
            3) A list of the components' weights.
        Each list has length 'n_components' and is sorted by non-increasing weight.
    """
    gmm_model = GaussianMixture(n_components=n_components)
    gmm_model.fit(cluster)

    # sort means by weight
    sorted_means = [m for _,m in sorted(zip(gmm_model.weights_,gmm_model.means_), reverse=True)]

    # find nearest pixel that actually exists in the cluster and replace
    modes = [0 for _ in range(len(sorted_means))]
    for i in range(len(sorted_means)):
        ind = np.argmin([np.linalg.norm(np.array(sorted_means[i])-p, ord=1) for p in cluster])
        modes[i] = cluster[ind]
    
    sorted_weights = sorted(gmm_model.weights_, reverse=True)

    return modes, sorted_means, sorted_weights


def gmm_label2rgb(image, labels_img, bg_color=(0,0,0), n_components=3):
    """ Fits each cluster to a GMM and paints it with the mode pixel value.

    The mode is determined by using the mean of the heaviest GMM mixture
    component for the cluster.

    Args:
        image: The image object.
        labels_img: The labels given to 'image' after segmentation.
        bg_color: Optional; the RGB color given to the background labels.
        n_components: Optional; the number of components in the GMM.

    Returns:
        A tuple containing
            1) The painted image.
            2) A list of GMM modes per cluster.
            3) A list of GMM means per cluster.
            4) A list of GMM component weights per cluster.
    """
    allowed_labels = list(set(labels_img.flatten().astype(int).tolist()))

    gmm_modes = []
    gmm_means = []
    gmm_weights = []
    H, W = image.shape[:2]
    for label in allowed_labels:
        if label == -1: continue
        cluster = np.array([image[i,j] for i in range(H) for j in range(W) if labels_img[i,j] == label])
        gmm = _cluster_gmm(cluster, n_components=n_components)
        gmm_modes.append(gmm[0])
        gmm_means.append(gmm[1])
        gmm_weights.append(gmm[2])

    # heaviest mode per cluster
    h_colors = [m[0] for m in gmm_modes]
    return _paint_img(image, labels_img, h_colors, bg_color=bg_color), gmm_modes, gmm_means, gmm_weights


def _paint_img(image, labels_img, label_colors, bg_color=(0,0,0)):
    """ Paints the segmented 'labels_img' using the 'label_colors'.

    Args:
        image: The image object.
        labels_img: The labels given to 'image' after segmentation.
        label_colors: A list of pixels assigned to each label.
        bg_color: Optional; the RGB color given to the background labels.

    Returns:
        The painted image.
    """
    bg_color = np.array(bg_color, dtype=np.uint8)
    painted_img = np.zeros(image.shape, dtype=np.uint8)

    H, W = image.shape[:2]
    for i in range(H):
        for j in range(W):
            label = labels_img[i,j]
            painted_img[i,j] = label_colors[label-1] if label != -1 else bg_color

    return painted_img


def vis_label_img(image, labels_img, median_seg, mode_seg, mask=None, use_mask=True):
    """ Displays the segmented image.

    Displays the original image, segmentation, overlaid segmentation,
    and mean-value segmentation.

    Args:
        image: The input image.
        labels_img: A 2D array of integer labels.
    """
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
    fig, ax = plt.subplots(nrows=2, ncols=3) #, sharex=True, sharey=True) #, figsize=(25, 8))
    _q = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    _t = ['Orig Image', 'Segmentation', 'Seg Overlay', 'Mean Seg', 'Median Seg', 'Mode Seg']
    for i, I in enumerate([image, pure_segs, over_segs, mean_segs, median_seg, mode_seg]):
        if not mask is None and use_mask:
            #print(I.dtype, I.dtype.type, I.dtype.kind, mask.dtype)
            if I.dtype.kind in ('u', 'i'): # Unsigned/signed integer cases
                maskp = 255 * mask.astype(I.dtype)
            elif I.dtype.kind == 'f': # Float case
                maskp = mask
            I = np.concatenate((I, maskp[:,:,np.newaxis]), axis = -1)
        j, k = _q[i]
        ax[j, k].imshow(I)
        ax[j, k].axis('off')
        ax[j, k].set_title(_t[i])
    plt.tight_layout()

def transition_matrix(L, normalize, print_M, keep_bg=False, verbose=False):
    """ Returns a transition count/probability matrix for 'L'.

    Each entry M_ij corresponds to the number of transitions from cluster i->j
    in 'L'.

    Args:
        L: An (M x N) matrix of integer labels in [1,K].
        normalize: If True, the matrix will be returned with the transition
          probabilties [0,1] for each entry. Otherwise, the transition counts
          will be returned.
        printM: If True, print the matrix before it is returned.
        keep_bg: Optional; If False, the transition counts from/to the background
          will be removed from the transition matrix. False by default.
        verbose: Optional; Print verbosely if True. False by default.

    Returns:
        A (K x K) transition count/probability matrix.
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
    if verbose: print('\tAssembled', len(all_tuples), 'edge pairs for transition estimation')
    if not keep_bg:
        if -1 in allowed_labels: allowed_labels.remove(-1) # 
        n_labels = len(allowed_labels)
        _ATN = len(all_tuples)
        all_tuples = [ elem for elem in all_tuples if not (-1 in elem) ]
        if verbose: 
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
    if print_M or verbose: 
        if verbose: print("Transition counts:", transition_counters)
        print('Labels:', allowed_labels)
        if verbose: 
            print('Column sums:', M.sum(1), '| Total:', np.tril(M,k=0).sum())
            extot = 2*H*W + W + H
            print('N_expected_transitions:', extot, '(unmasked: %d)' % (extot - n_masked - H - W))
        print('Transition matrix')
        print(M)
    return M, transition_counters

def segment(image, method, mask, gc_compactness=10.0, gc_n_segments=300, gc_slic_sigma=0.0):
    """ Performs spatially aware image segmentation.

    The image is segmented using k-means clustering followed by a
    normalized graph cut.
    
    Args:
        image: The input image.
        mask: A binary alpha mask.
        gc_compactness: Optional; Weight given to space proximity (as opposed to 
          colour proximity). Higher values result in higher superpixel compactness.
          10.0 by default.
        gc_n_segments: Optional; Approximate number of clusters created
          during the k-means clustering. 300 by default.
        gc_slic_sigma: Optional; Width of the Gaussian smoothing kernel applied
          to each colour channel. If gc_slic_sigma is 0, no pre-processing
          is done. 0.0 by default.

    Returns:
        A 2D array of integer labels corresponding to the original image. Background pixels
        are given a label of -1.
    """
    #if not (mask is None):
    #    H, W, C = image.shape
    #    image = np.concatenate( (image, mask.reshape(H,W,1)), axis=2)
    image = send_mask_to_minus_one(image, mask)
    if method == 'graph_cuts':
        labels1 = segmentation.slic(image, compactness=gc_compactness,
                                    n_segments=gc_n_segments, 
                                    sigma=gc_slic_sigma) #, compactness=gc_compactness)
        out1    = color.label2rgb(labels1, image, kind='avg')
        g       = graph.rag_mean_color(image, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
        labels2 = send_mask_to_minus_one(labels2, mask)
        return labels2

def gauss_filter(img, blur_sigma):
    """ Performs Gaussian blurring on 'img' with standard deviation 'sigma'.

    Args:
        image: The input image.
        sigma: The standard deviation of Gaussian kernel.

    Returns:
        The blurred image.
    """
    assert blur_sigma >= 0.0, "Untenable blur kernel width"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_ubyte( gaussian_blur(img, blur_sigma) )
    return img

def send_mask_to_minus_one(image, mask):
    image[ mask <= 0 ] = -1 
    return image
