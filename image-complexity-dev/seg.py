import os, sys, module.utils as utils, argparse, sklearn, skimage, numpy as np, numpy.random as npr
import warnings, matplotlib.pyplot as plt, csv, random
from skimage import data, segmentation, color
from skimage.color import rgb2hsv
from skimage.future import graph
from sklearn import cluster
from scipy import interpolate
from module.segmentation import *


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
    parser.add_argument('--verbose', dest='verbose', action='store_true',
        help='Whether to print verbosely while running')
    parser.add_argument('--rseed', type=int, default=None,
        help='Specify a fixed random seed to use for this run.')
    ### Image preprocessing
    parser.add_argument('--resize', type=float, default=1.0,
        help='Specify scalar resizing. E.g., 0.5 halves the image size; 2 doubles it. (default: 1.0)')
    parser.add_argument('--blur', type=float, default=0.0,
        help='Specify Gaussian blur standard deviation applied to the image (default: 0.0)')
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
    parser.add_argument('--seg_mean_stats_output_file', default=None,
        help='Specifies output file to which segment statistics with cluster-mean values should be written')
    parser.add_argument('--seg_median_stats_output_file', default=None,
        help='Specifies output file to which segment statistics with cluster-median values should be written')
    parser.add_argument('--seg_mode_stats_output_file', default=None,
        help='Specifies output file to which segment statistics with cluster-mode values should be written.' +
              ' The modes are computed using GMM mixture component means.')
    parser.add_argument('--cluster_number_file', default=None,
        help='Specifies output file to which the number of estimated clusters per image should be written')
    ### Output image files
    parser.add_argument('--write_mean_segs', action='store_true', dest='write_mean_segs',
        help='Writes the segmented image(s) with cluster-mean values. Requires mean_seg_output_dir.')
    parser.add_argument('--mean_seg_output_dir', default=None,
        help='Specifies the folder to which saved mean segment images must be written')
    parser.add_argument('--write_median_segs', action='store_true', dest='write_median_segs',
        help='Writes the segmented image(s) with cluster-median values. Requires median_seg_output_dir.')
    parser.add_argument('--median_seg_output_dir', default=None,
        help='Specifies the folder to which saved median segment images must be written')
    parser.add_argument('--write_mode_segs', action='store_true', dest='write_mode_segs',
        help='Writes the segmented image(s) with cluster-mode values. Requires mode_seg_output_dir.' +
              ' The modes are computed using GMM mixture component means.')
    parser.add_argument('--mode_seg_output_dir', default=None,
        help='Specifies the folder to which saved mode segment images must be written' +
              ' The modes are computed using GMM mixture component means.')
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

    # Set random seed state
    if not args.rseed is None:
        # Note that this is the legacy approach, not the ideal one, for numpy
        if args.verbose: print('Setting random seed to', args.rseed)
        np.random.seed(args.rseed)
        random.seed(args.rseed)

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
    if os.path.isdir(args.input):
        usables = [ '.jpg', '.png' ]
        usables = list(set( usables + [ b.upper() for b in usables ] + 
                                      [ b.lower() for b in usables ] ))
        _checker = lambda k: any( k.endswith(yy) for yy in usables )
        targets = [ os.path.join(args.input, f) 
                    for f in os.listdir(args.input) if _checker(f) ]
        for t in targets: 
            file_outputs[t] = main_helper(t, args)
    elif os.path.isfile(args.input):
        file_outputs[args.input] = main_helper(args.input, args)
    else:
        raise ValueError('Non-existent target input ' + args.input)

    # Write seg info results
    if not (args.seg_mean_stats_output_file is None):
        if args.verbose: print('Writing seg file to', args.seg_mean_stats_output_file)
        with open(args.seg_mean_stats_output_file, "w") as _fh:
            #_fh.write("file,cluster_ind,mu_C_1,mu_C_2,mu_C_3,n_member_pixels,percent_member_pixels\n")
            _fh.write("image,cluster,RGB_R,RGB_G,RGB_B,HSV_H,HSV_S,HSV_V,LAB_L,LAB_A,LAB_B,frequency,percent\n")
            for key in file_outputs.keys(): # For each file
                tmat, D_img = file_outputs[key]
                Ds = D_img['cluster_info']['cluster_stats']
                clines = []
                for cluster_label in Ds.keys(): # For each cluster
                    D = Ds[cluster_label]
                    clines.append( [ key, D['label_number'], 
                               D['mean_C1'], D['mean_C2'], D['mean_C3'], 
                               D['mean_C1_hsv'], D['mean_C2_hsv'], D['mean_C3_hsv'], 
                               D['mean_C1_lab'], D['mean_C2_lab'], D['mean_C3_lab'],
                               D['n_member_pixels'], 
                               D['percent_member_pixels'] ] ) # ) )
                clines.sort(key = lambda a: a[-1], reverse = True)
                clines = [ ",".join( map(str, a) ) for a in clines ]
                for line in clines: _fh.write(line + '\n')

    # Write median seg info results
    if not (args.seg_median_stats_output_file is None):
        if args.verbose: print('Writing median seg file to', args.seg_median_stats_output_file)
        with open(args.seg_median_stats_output_file, "w") as _fh:
            _fh.write("image,cluster,RGB_R,RGB_G,RGB_B,HSV_H,HSV_S,HSV_V,LAB_L,LAB_A,LAB_B,frequency,percent\n")
            for key in file_outputs.keys(): # For each file
                tmat, D_img = file_outputs[key]
                Ds = D_img['cluster_info']['median_cluster_stats']
                clines = []
                for cluster_label in Ds.keys(): # For each cluster
                    D = Ds[cluster_label]
                    clines.append( [ key, D['label_number'], 
                               D['median_C1'], D['median_C2'], D['median_C3'], 
                               D['median_C1_hsv'], D['median_C2_hsv'], D['median_C3_hsv'], 
                               D['median_C1_lab'], D['median_C2_lab'], D['median_C3_lab'],
                               D['n_member_pixels'], 
                               D['percent_member_pixels'] ] ) # ) )
                clines.sort(key = lambda a: a[-1], reverse = True)
                clines = [ ",".join( map(str, a) ) for a in clines ]
                for line in clines: _fh.write(line + '\n')

    # Write mode seg info results
    _N = 3 # number of components in GMM
    if not (args.seg_mode_stats_output_file is None):
        if args.verbose: print('Writing mode seg file to', args.seg_mode_stats_output_file)
        with open(args.seg_mode_stats_output_file, "w") as _fh:
            _fh.write("image,cluster,")
            for i in range(1, _N+1):
                _fh.write((f'gmm{i}_mean_R,gmm{i}_mean_G,gmm{i}_mean_B,gmm{i}_weight,'
                f'mode{i}_RGB_R,mode{i}_RGB_G,mode{i}_RGB_B,mode{i}_HSV_H,'
                f'mode{i}_HSV_S,mode{i}_HSV_V,mode{i}_LAB_L,mode{i}_LAB_A,mode{i}_LAB_B,'))
            _fh.write(",frequency,percent\n")
            for key in file_outputs.keys(): # For each file
                tmat, D_img = file_outputs[key]
                Ds = D_img['cluster_info']['mode_cluster_stats']
                clines = []
                for cluster_label in Ds.keys(): # For each cluster
                    D = Ds[cluster_label]
                    _row = [key, D['label_number']]
                    for i in range(1, _N+1): # For each mode
                        _row.extend([
                            D[f'gmm{i}_C1_mean'], D[f'gmm{i}_C2_mean'], D[f'gmm{i}_C3_mean'], D[f'gmm{i}_weight'],
                            D[f'mode{i}_C1'], D[f'mode{i}_C2'], D[f'mode{i}_C3'],
                            D[f'mode{i}_C1_hsv'], D[f'mode{i}_C2_hsv'], D[f'mode{i}_C3_hsv'],
                            D[f'mode{i}_C1_lab'], D[f'mode{i}_C2_lab'], D[f'mode{i}_C3_lab']
                        ])
                    _row.extend([D['n_member_pixels'], D['percent_member_pixels']])
                    clines.append(_row)
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

    img = skimage.io.imread(img_path)
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
        if args.verbose: print("Resized dims:", img.shape)
    
    # Handle masking (alpha transparency)
    mask = None
    if n_channels == 4:
        a = img[:, :, 3]
        img = img[:, :, :3]
        mask = a.copy()
        mask[ a >  128 ] = 1 # 255
        mask[ a <= 128 ] = 0
        # Zero out the background
        if not args.ignore_alpha:
            img *= mask[:,:,np.newaxis]

    # Default mask: non-transparent alpha channel
    H, W, C = img.shape
    if (mask is None) or args.ignore_alpha: 
        mask = np.ones( (H,W) )

    # Gaussian filter if desired
    if args.blur > 1e-8:
        if args.verbose: print("\tBlurring with sigma =", args.blur)
        img = gauss_filter(img, args.blur)

    if args.verbose: 
        print('\tShape (%d,%d,%d)\n\tmin/max vals:' % (H,W,C), 
              img.min(0).min(0), img.max(0).max(0))
        print('\tNum masked values:', H*W - (mask > 0).sum(), "/", H*W)

    # Compute labelling -> merge clusters -> reorder labels
    label_image = reorder_label_img( 
                        merge_smaller_clusters(img, mask, label(img, mask, args.labeller, clustering_colour_space=args.clustering_colour_space,
                                                                dbscan_eps=args.dbscan_eps, dbscan_min_neb_size=args.dbscan_min_neb_size,
                                                                kmeans_k=args.kmeans_k, kmeans_auto_bounds=args.kmeans_auto_bounds,
                                                                kmeans_auto_crit=args.kmeans_auto_crit, gc_compactness=args.gc_compactness,
                                                                gc_n_segments=args.gc_n_segments, gc_slic_sigma=args.gc_slic_sigma,
                                                                verbose=args.verbose),
                                               merge_small_clusters_method=args.merge_small_clusters_method,
                                               small_cluster_merging_kdep_param=args.small_cluster_merging_kdep_param,
                                               fixed_cluster_size_merging_threshold=args.fixed_cluster_size_merging_threshold,
                                               small_cluster_merging_dynamic_k=args.small_cluster_merging_dynamic_k,
                                               verbose=args.verbose),
                                    verbose=args.verbose)
    mean_seg_img = color.label2rgb(label_image, image = img, bg_label = -1, 
                                    bg_color = (0.0, 0.0, 0.0), kind = 'avg')
    median_seg_img, medians = median_label2rgb(img, label_image)
    mode_seg_img, modes, gmm_means, gmm_weights = gmm_label2rgb(img, label_image)

    _write_img = [args.write_mean_segs, args.write_median_segs, args.write_mode_segs]
    _img = [mean_seg_img, median_seg_img, mode_seg_img]
    _output_dir = [args.mean_seg_output_dir, args.median_seg_output_dir, args.mode_seg_output_dir]
    _type = ['mean', 'median', 'mode']

    # Write mean/median/mode-cluster-valued image out if desired
    for i in range(len(_write_img)):
        if _write_img[i]:
            int_mask = (mask*255).reshape(H,W,1)
            segs4 = np.concatenate( (_img[i], int_mask), axis = 2 )
            if not os.path.isdir(_output_dir[i]):
                os.makedirs(_output_dir[i])
            cfname = os.path.join(_output_dir[i], 
                                img_file_basename_extless + f".{_type[i]}_seg.png")
            if args.verbose: print(f'\tSaving {_type[i]} seg image to', cfname)
            skimage.io.imsave(fname = cfname, arr = segs4)

    ### Compute transition matrix and other stats ###
    M = transition_matrix(label_image, args.normalize_matrix, 
            print_M = (not args.no_print_transitions), keep_bg=args.keep_bg, verbose=args.verbose )
    img_stats = label_img_to_stats(img, mask, label_image, mean_seg_img, medians=medians, modes=modes,
                                   gmm_means=gmm_means, gmm_weights=gmm_weights, verbose=args.verbose)

    ### Visualization ###
    if args.display: vis_label_img(img, label_image, median_seg_img, mode_seg_img, mask)
    plt.show()

    if args.verbose: print('Finished processing', img_path)
    return M, img_stats

#-------------------------#
if __name__ == '__main__':
    main()
#-------------------------#


#
