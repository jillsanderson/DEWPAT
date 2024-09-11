""" Implmentation of AcuityView described in Caves and Johnsen, 2017.
"""

import skimage
import numpy as np
from scipy import fftpack as fp
from .utils import load_helper


def acuity_view(img_path, w, d, R=0.08):
    """ Outputs an image as it would appear to an animal.

    Follows the AcuityView library implementation in R
    https://github.com/eleanorcaves/AcuityView

    Notes:
        1) The original implementation required the image to be a
        square with each side being a power of 2 pixels. This function
        will accept any image dimensions.

        2) The fftw2d function in R's fftwtools library does not return
        the original matrix when the FFT and iFFT is applied to a matrix;
        each element is multiplied by the width and height of the matrix.
        The original implementation divided the fourier images by the
        width of the image but this is not done here.

    Args:
        img_path: An image path.
        w: The real width of the image. Must be in the same units
            as the viewing distance 'd'.
        d: The viewing distance. Must be in the same units as the
            real width 'w'.
        R: Optional; the minimum resolvable angle of the viewer.

    Returns:
        The blurred image.
    """
    img = load_helper(img_path, verbose=False)[0]
    img = srgb2linear(img)

    # angular width of image
    a = np.rad2deg(2 * np.arctan(w / d / 2))

    h, w = img.shape[:2]
    K = MTF(h, w, a, R)

    fourier_images = [ fp.fft2(img[:,:,i]) for i in range(3) ]

    shifted_fourier = np.array([ np.fft.fftshift(fourier_image) * K  for fourier_image in fourier_images ])
    inverse_fourier = np.array([ np.abs(fp.ifft2(shifted_image)) for shifted_image in shifted_fourier ])

    blurred_image = linear2srgb(np.dstack(inverse_fourier))

    # convert to uint8 to suppress warning
    blurred_image = (blurred_image*255).astype(np.uint8)
    return blurred_image
    

def srgb2linear(img):
    """ sRGB to linear RGB colour space conversion.

    Args:
        img: The sRGB image object.

    Returns:
        The linear RGB image.
    """

    norm_img = img / 255

    # rescale image to [0, 1]
    for i in range(3):
        channel = norm_img[:,:,i]
        _max = np.max(channel)
        _min = np.min(channel)

        norm_img[:,:,i] = (channel - _min) / (_max - _min)

    f = lambda x: x / 12.92 if x < 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    return np.vectorize(f)(norm_img)


def linear2srgb(img):
    """ Linear RGB to sRGB colour space conversion.

    Args:
        img: The linear RGB image object.

    Returns:
        The sRGB image.
    """
    f = lambda x : x * 12.92 if x < 0.0031308 else ((1.055) * x**(1/2.4)) - 0.055
    return np.vectorize(f)(img)


def MTF(h, w, a, R):
    """ Returns a blur matrix.

    Args:
        h: The height of the image in pixels.
        w: The width of the image in pixels.
        a: The angular width of the image.
        R: The minimum resolvable angle of the viewer.
    """

    center_h = h // 2
    center_w = w // 2

    blur = np.zeros(shape=(h, w))
    a = w / a

    for i in range(h):
        for j in range(w):
            x = i - center_h
            y = j - center_w

            freq = np.round(np.sqrt(x**2 + y**2)) / w * a
            blur[i, j] = np.exp(-3.56 * (freq * R)**2)
    
    return blur
