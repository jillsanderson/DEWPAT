# General overview statement

DEWPAT currently exists as a series of Python scripts divided between two branches. Note that both branches should technically contain all of the scripts, but for best and most up to date functionality you should use the script from the branch designated below.

### master branch

- img_complexity.py: main script that includes functionality to compute and visualize complexity across both sRGB and multispectral images. Note that if using mspec images, need to run --mspec along with other arguments

- vis.py: includes visualization options (including 1D colour histogram) for sRGB images only


### dev branch

- seg.py: includes functionality to segment, visualize, and output colour statistics for sRGB images only

- preprocess.py: includes functionality to blur both sRGB and multispectral images to model visual acuity and viewing distance using AcuityView (Caves & Johnsen, 2017).

# Requirements

Requires: scikit-image, numpy, matplotlib, scikit-learn, and scipy.
An easy way to obtain them is via the conda environment specified in `env.yaml`.

Create environment: `conda env create -f env.yaml`.

Activate environment: `source activate imcom` or `conda activate imcom`.
(On Windows, just `activate imcom`).

# General usage

The general usage is as follows: `python <script_name.py> <file.extension or input folder> --usage_specific_argument <output.csv>`

We note that if users wish to run scripts using default algorithm parameters (e.g., patch sizes, bin numbers, etc.) then they can simply run commands in the terminal. If users wish to change default parameters, they first need to find and edit the parameters in the scripts themselves. See the main text for details about default parameters.


# img_complexity.py

This script contains implementations of several simple measures of image complexity,
including ones based on frequency content, information entropy, spatial derivatives, and differences in local statistical moments.
Some basic visualization methods are also present.


## Usage

Simply run `python img_complexity.py <input>`, where `<input>` can be a folder of images or a single image. 

Run `python img_complexity.py --help` to print detailed usage help from the script.

#### Intermediate Visualizations

You can display intermediate results/images for computations that support it.
For instance, `--show_local_ents` displays a color map with local estimated patch entropies over the image.


#### Alpha Channel Masking

Most measures are able to account for the presence of an alpha channel mask, allowing you to run the measures on a ROI rather than the entire image. A simple way to include an alpha mask is to remove the background beforehand (i.e., using photoshop), and then save the image as a .png. As a rule of thumb, png images always have an alpha color channel (this is why png images can have transparent backgrounds).
In the patch-based estimators, any patch with a masked pixel is ignored. 
The alpha channel can be ignored with the flag `--ignore_alpha`.


#### Preprocessing

There are a few preprocessing options within img_complexity.py (not including the acuity blurring done through preprocess.py in the dev branch):

- **Greyscale**: passing `--greyscale human` will convert the image to greyscale via channel weightings based on human perceptual weightings, while `--greyscale avg` will use uniform weights (average over channels). Important: note that the determinant of the covariance becomes degenerate (singular) for a scalar image (since the channels have no covariance in this case); therefore *trace* rather than *determinant* is taken in those cases.

- **Resizing**: for instance, passing `--resize 0.5` halves the dimensions of the input image for all complexities. Note that passing `--resize` overrides the value for `--emd_downscaling`, which applies only to the pairwise Wasserstein metric.

- **Blurring**: passing `--blur 5`, for instance, will low-pass filter the image with a Gaussian of standard deviation 5. Note that resizing happens *before* blurring, so choose the standard deviation with this in mind.

- **Gradient Image**: The metrics can either be computed on an input image $`I`$ or on the per-channel *gradient magnitude image*
$`I_G=||\nabla I||_2`$ of that input (or both). Use `--use_grad_only` to use the gradient image and `--use_grad_too` to compute the measures on *both* the original input and its gradient image.

#### Examples

- Compute all the complexity measures for a single input image and output the data to a .csv file:

  `python img_complexity.py eg.png >output.csv

- Compute all the complexity measures for all images in a folder and output the data to a .csv file:

  `python img_complexity.py folder >output.csv

- Compute all the complexity measures for a single input image, as well as visualizing the contributions of each pixel for the local pixelwise entropy measure:

  `python img_complexity.py eg.png --show_local_covars`

- Compute only the patch-wise differential entropy measure and the weighed Fourier measure, on both the input image and its gradient image, for every image in the input folder:

  `python img_complexity.py eg_folder --diff_shannon_entropy_patches --weighted_fourier --use_grad_too`

- Compute the pairwise Wasserstein distance among patches for a single input image using the Sinkhorn approximation, while also visualizing the patches used in the computation and printing timing information:

  `python img_complexity.py example.jpg --timing --pairwise_emd --show_emd_intermeds --sinkhorn_emd`

- Compute all measures (except pairwise EMD) on images in `folder`, with each resized to 65% of original size, converted to greyscale (by averaging over channels), and having applied a low-pass Gaussian blurring with $`\sigma=4`$, as well as display the resulting preprocessed images:

  `python img_complexity.py folder --blur 4 --greyscale avg --resize 0.65 --show_img`

- List all command flags and options of the script, along with descriptions of each:

  `python img_complexity.py --help`

## Complexity Measures

We briefly provide a basic description of the measures.
They may be slightly altered (e.g., monotonically transformed or scaled) to give more aesthetically pleasing values.
See the code for precise computational details.
Notation: image $`I`$ with channels $`c\in C`$, set of patches $`P`$ (per channel, $`P_c`$), set of pixels $`V`$ (per channel, $`V_c`$), and empirical covariance matrix $`\widehat{C}`$.

### Pixel-wise Discrete Entropy

Computes the discrete Shannon entropy over individual pixel values across the image, written
```math
\mathcal{H}_{d,v}(I) = \frac{-1}{|C|} \sum_{c\in C} \sum_{s\in V_c} p(s) \log p(s)
```

### Average Local Pixel-Wise Entropies

Computes the mean discrete Shannon entropy within local image patches, via
```math
\mathcal{H}_{d,p}(I) = \frac{-1}{|C|} \sum_{c\in C} \frac{1}{|P_c|} \sum_{\zeta\in P_c} \sum_{v\in \zeta}  p(v) \log p(v)
```

### Frequency-Weighted Mean Fourier Coefficient

Computes the weighted average of the Fourier coefficient values, with weights based on the associated frequency value, such that
```math
\mathcal{F}_w(I) = \frac{1}{Z_\gamma} \sum_{\psi_x,\,\psi_y\in \Psi} \gamma(\psi_x,\psi_y)\, \mathcal{I}_F(\psi_x,\psi_y),
\,\;\text{ where}\;\;
\mathcal{I}_F = \frac{1}{|C|}\sum_{c\in C} \log \left| \mathfrak{F}[c] \right|
\;\;\text{and}\;\;
Z_\gamma = \sum_{\psi_x,\,\psi_y\in \Psi} \gamma(\psi_x,\psi_y),
```
denoting $`\mathfrak{F}[c]`$ as the Fourier transform of the single-channel image $`c`$, $`\Psi`$ as the set of frequency space coordinates, and $`\gamma(x,y)`$ is a form of distance function (e.g., $`\gamma(x,y)=||x-y||_\ell`$). By default, alpha-masked values are filled with the mean unmasked pixel value.

### Average Local Patch Covariances

Estimates the mean local intra-patch covariance over the image, written
```math
\mathcal{C}_L(I) = \frac{1}{|P|} \sum_{p\in P} \log\left( \det\left(\widehat{C}(p)\right) + 1 \right)
```
Note that trace instead of determinant is used in the greyscale case.

### Average Gradient Magnitude

Computes the mean value of the per-channel gradient magnitude over the image; i.e.,
```math
\mathcal{G}(I) = \frac{1}{|C|\,|V_c|} \sum_{c\in C} \sum_{s\in V_c} ||\nabla I(s)||_2
```

### Pixel-wise Differential Entropy

Estimates the differential Shannon entropy of the continuous-space vector-valued pixel distribution of the image.
This can be written
```math
\mathcal{H}_{c,v}(I) = -\int_V p(v) \log p(v)\, dv
```

### Patch-wise Global Differential Entropy

Estimates the differential entropy of the distribution of patches over the images, where each multi-channel patch is unfolded into a single continuous vector, written
```math
\mathcal{H}_{c,p}(I) = -\int_P p(\xi) \log p(\xi)\, d\xi
```

### Global Patch Covariance

Computes the log-determinant of the global covariance matrix over patches in the image, where again each multi-channel patch is unfolded into a single vector; i.e.,
```math
\mathcal{C}_{G}(I) = \log\left( \det\left( \widehat{C}(P) \right) \right)
                                   max_eps = np.inf,
```
Note that trace instead of determinant is used in the greyscale case.

### Pairwise Distance between Patch Means

Computes the average pairwise distance between the first moments (means) of the patches across the image:
```math
\mathcal{D}_{\mathcal{M},1}(I) = \frac{1}{|C|\,|P|^2} \sum_{p_i,p_j\in P}
|| \widehat{\mu}(p_i) - \widehat{\mu}(p_j) ||_2
```
where $`\widehat{\mu}(p)\in\mathbb{R}^3`$ is the mean pixel value over patch $`p`$.

By default, this method utilizes *non-overlapping* patches (as do the other divergences).

### Pairwise Distance between Patch Moments

This measure is similar to the one just above, except that it considers the second moment (the covariance) as well:
```math
\mathcal{D}_{\mathcal{M},2}(I) = \frac{1}{|P|^2} \sum_{p_i,p_j\in P}
\frac{\gamma_\mu}{|C|}   || \widehat{\mu}(p_i) - \widehat{\mu}(p_j) ||_2 +
\frac{\gamma_C}{|C|^2} || \widehat{\Sigma}(p_i) - \widehat{\Sigma}(p_j) ||_{1,1/2}
```
where $`\widehat{\Sigma}(p)\in\mathbb{R}^{3\times 3}`$ is the covariance matrix of pixel values over the patch $`p`$ and $`||M||_{1,1/2} = \sqrt{\sum_{k,\ell} |M_{k\ell}| }`$.

We can also compute the pairwise distance based on these moments, using distributional divergences. 
These correspond to Gaussian (maximum entropy) assumptions on the pixel distributions per patch 
    (i.e., the minimum information assumption given the moments).
This includes options for the Jeffrey's (symmetric KL) divergence, the Wasserstein-2 metric, the squared Hellinger distance, the Bhattacharyya distance, and the Forstner-Moonen Abou-Moustafa-Torres-Ferries (FM-ATF) density metric (see "Designing a Metric for the Difference Between Gaussian Densities"). 
See the `--pwg_*` options, and also [this readme](module/README.md).

By default, this method utilizes *non-overlapping* patches as well and sets $`\gamma_C=\gamma_\mu=1`$.

### DWT coefficients

This measure is based on how much information is contained in the largest DWT detail coefficients $`D`$ extracted from an image $`I`$ of size $`M \times N`$.

```math
S_{DWT} = \frac{1}{MN}\sum_{d_i \in D}|d_i|
```

By default, we define $`D`$ as the set of the largest (in absolute value) 1% horizontal, vertical, and diagonal coefficients. 4 levels of DWT are applied using the Haar mother wavelet.

# vis.py

The file `vis.py` includes several visualization capabilities, for understanding pixel value distributions. These options include:

- `--show_img`: displays the original image (with its alpha mask if present).

- `--hist_rgb_1d`: displays the marginal distributions of each.

- `--hist_rgb_3d`: displays a 3D interactive histogram of pixel colour values.

- `--scatter_densities`: displays a 3D scatter plot of a random subset of pixel values, with their projected 2D marginal densities.

- `--manual_unfolded_1d`: displays a 1D histogram of the pixel values with respect to some arbitrarily defined ordering of 3D colourspace on the 1D real line, via a manually chosen colormap
$`C:[0,1]\rightarrow[0,1]^3`$.

- `--projected_pixels`: displays a random subset of coloured pixel points projected in their PCA subspace.

- `--show_hsv`: displays the HSV decomposition

- `--hist_hsv_polar`: shows the polar plot of the cyclic hue value histogram

- `--show_twilight_img`: displays the image, scalarized according to the cyclic *twilight* colourmap

- `--hist_twilight_polar`: shows the polar plot of the cyclic *twilight* colourmap value histogram

- `--all`: displays all the aforementioned visualizations.

It can also write the histogram values of the 1D ("manually unfolded") histogram to a file
via `--write_1d_histo_vals --output_file histo_output.csv`.

For complete details, run `python vis.py --help`.

NOTE: if you want to change the bin numbers for the histogram, find the nbins=75 in the vis.py script and change the number to the desired bin number. The manual colour bar includes 14 seperate colours, so we recommend this as the minimum. 

# Clustering and Segmentation
Note that for most up-to-date functionality, users should use seg.py in the dev branch.

The file `seg.py` includes some clustering/segmentation capabilities in pixel space. 
It also includes transition matrix analysis calculations.

Run `python seg.py --help` gives a complete list of possible options. 
I also recommend running with `--verbose` so you know it is acting as intended.

### Examples
- Run the clustering under default settings, printing verbosely, and display the resulting segmented image at the end:

```bash
python seg.py <target> --display --verbose
```

- Run the clustering in CIE-LAB colour space, merge small clusters with their closest counterparts in colour space (with a fixed threshold of 0.05), and write out the number of clusters per image:

```bash
python seg.py <target_directory> --display --verbose \
              --clustering_colour_space lab \
              --cluster_number_file cn.csv \
              --merge_small_clusters_method fixed \
              --fixed_cluster_size_merging_threshold 0.05
```

- Segments the image (after resizing and blurring) via graph cuts with a specified compactness and superpixel initialization parameters:  
  
```bash
python seg.py <target> --display --verbose \
              --labeller graph_cuts --gc_compactness 20 \
              --resize 1.0 --blur 0.25 --gc_n_segments 500
```

- Segments the image with k-means using `k` values from `k.csv`, writes the mean segmented images to `segs_dir`, and writes a csv of cluster data to `cluster_data.csv`:

```bash
python seg.py <target> --kmeans_k_file_list k.csv --verbose \
              --write_mean_segs --mean_seg_output_dir segs_dir \
              --seg_mean_stats_output_file cluster_data.csv
```


- Segments the image with k-means using `k` value of 4, writes the median segmented images to `med_segs_dir`, writes a csv of cluster data to `cluster_med_data.csv`, and does not print the transition matrix analysis calculations:

```bash
python seg.py <target> --labeller kmeans --kmeans_k 4 --verbose \
              --write_median_segs --median_seg_output_dir med_segs_dir \
              --seg_median_stats_output_file med_segs.csv \
              --no_print_transitions
```

# preprocess.py (modeling visual acuity)

We provide a simple method of accounting for visual acuity blurring effects, based on [AcuityView](https://github.com/eleanorcaves/AcuityView), induced by the real sizes and distances of the objects and visual systems involved. Call `python preprocess.py --help` for usage help.

Usage is as follows `python preprocess.py <input image or folder> <.csv> <output.csv> --verbose` See Supporting Info doc for exmaple of .csv file format.

## Example

- Blur images in folder images_to_blur according to the data in the acuity.csv file (which includes the size of the objects in the images, the viewing distance in the same units as the size, and the visual acuity in terms of minmum resolvable angel (MRA), output the blurred images to the folder blur_output, and print all of this verbosely:

  `python preprocess.py images_to_blur acuity.csv blur_output --verbose`

- Segments the image with k-means using
`python preprocess.py blur_full blur_full.csv blur_full_out --verbose`

## Acknowledgements

The differential entropy measures utilize the *Non-parametric Entropy Estimation Toolbox* by Greg Ver Steeg.
See the [**NPEET** Github Repo](https://github.com/gregversteeg/NPEET) (MIT licensed).
It implements the approach in Kraskov et al, 2004, *Estimating Mutual Information*.
