"""
Implements several functions useful for visualization of images, particularly color distributions.
"""

import os, sys, numpy as np, skimage, argparse, scipy, matplotlib
from skimage import io, color as skcolor
import matplotlib.pyplot as plt
from module.utils import load_helper
from mpl_toolkits.mplot3d import Axes3D, axes3d
from module.visualization import *


def main_single_display(args):
    assert os.path.exists(args.input), "Input " + args.input + " does not exist."
    img, R, G, B, orig_mask = load_helper(args.input, blur_sigma=args.blur)

    #---------------------------------------------------------------------------------#

    ### Show the original image ###
    if args.show_img: display_orig(img, orig_mask, args.input)
    ### Show HSV decomposition ###
    if args.show_hsv: display_hsv(img, orig_mask)
    ### Polar Hue histogram from HSV ###
    if args.hist_hsv_polar: plot_polar_hsv(img, orig_mask)
    ### Show Scalar colormap index image ###
    if args.show_twilight_img: # TODO other maps
        plot_colour_mapped_scalar_image(img, orig_mask, 'twilight')
    ### Polar Colour histogram from named colormaps ###
    if args.hist_twilight_polar: # TODO other maps
        plot_polar_generic(img, orig_mask, 'twilight')
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



#
