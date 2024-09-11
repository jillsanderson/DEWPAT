import numpy as np, pywt
import scipy.ndimage.morphology as morpho
import scipy.misc as deprecImProc
import skimage.transform as imProc
from matplotlib import cm
import matplotlib.pyplot as plt
import PIL


def evalComplexity(im,mask,thrPercentile=99,levels=4,mWavelet='haar'):
    # Evaluates the complexity of a dewlap image using DWTs
    # wavelet variable defined which "wavelet" to use. "Haar" wavelet is usually the one used by default.
    if im.max()> 1:
        im = im.astype(float)/255

    (cA, cH, cV, cD) = computeImDWT(im, mask, levels,mWavelet)

    # Threshold detail coefficients to keep only the X-th most important ones
    subsetH = sampleCoeffs(cH, thrPercentile)
    subsetV = sampleCoeffs(cV, thrPercentile)
    subsetD = sampleCoeffs(cD, thrPercentile)
    C = np.append(np.append(subsetH,subsetV),subsetD)
    # Compute score (sum of detail coefficients)
    score = np.sum(np.absolute(C))/im[:,:,0].size

    return score


def visualize(im, mask, levels=4, mWavelet='haar', show=True):
    # Generates a pyramid-like image with every DWT coefficients (see wikipedia DWT page for an example)
    # Format:
    # [approx, horizontal]
    # [vertical, diagonal]

    (cA,cH,cV,cD) = computeImDWT(im,mask,levels,mWavelet)

    cA = np.divide(cA,2**levels)
    # Build image, starting at the finest scale
    for j in range(levels,0,-1):
        i = j-1
        # Normalize coefficients for simpler vis
        cH[i] = normalizeCoeff(cH[i])
        cH[i] = np.array(PIL.Image.fromarray(np.uint8(cm.bone(cH[i]) * 255)))[:,:,0:3].astype(float)/255
        cV[i] = normalizeCoeff(cV[i])
        cV[i] = np.array(PIL.Image.fromarray(np.uint8(cm.bone(cV[i]) * 255)))[:, :, 0:3].astype(float) / 255
        cD[i] = normalizeCoeff(cD[i])
        cD[i] = np.array(PIL.Image.fromarray(np.uint8(cm.bone(cD[i]) * 255)))[:, :, 0:3].astype(float) / 255

        if j==levels:
            catTop = np.concatenate((cA, cH[i]), 1)
        else:
            visTemp = imProc.resize(visTemp,cH[i].shape,0)

            catTop = np.concatenate((visTemp,cH[i]),1)

        catBottom = np.concatenate((cV[i],cD[i]),1)
        visTemp = np.concatenate((catTop,catBottom),0)

    visOut = visTemp
    fig, ax = plt.subplots()
    #print(visOut.max())
    #print(visOut.min())
    ax.set_title('Wavelets')
    fig.canvas.set_window_title('Wavelets')
    #fig.suptitle("Wavelets")
    # Clip to avoid annoying warning
    visOut = np.clip(visOut, 0, 1)
    imgPlot = plt.imshow(visOut)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    if show:
        plt.show()
    return visOut

def _resize_loc(img, new_size):
    from PIL import Image
    return np.array(
                Image.fromarray(
                    (img*255).astype(np.uint8)
                ).resize(new_size)
            ).astype(float) / 255


def computeImDWT(im,mask,levels,waveletType):

    numChannels = im.shape[2]
    # Compute DWT of first color channel
    for k in range(0,numChannels):
        (cAt, cHt, cVt, cDt) = computeImDWTsingleChannel(im[:, :, k], levels, waveletType)
        if mask is not None:
            for i in range(0, levels):
                tempMask = _resize_loc(mask, reversed(cHt[i].shape)) > 0 
                #tempMask = deprecImProc.imresize(mask, cHt[i].shape) >0
                
                tempMask = morpho.binary_erosion(tempMask, morpho.iterate_structure(morpho.generate_binary_structure(2, 2),2))

                cHt[i] *= tempMask
                cVt[i] *= tempMask
                cDt[i] *= tempMask

        if k==0:
            cA, cH, cV, cD = cAt,cHt, cVt, cDt
        else:
            cA = np.dstack((cA, cAt))
            for j in range(levels):
                cH[j] = np.dstack((cH[j], cHt[j]))
                cV[j] = np.dstack((cV[j], cVt[j]))
                cD[j] = np.dstack((cD[j], cDt[j]))

    return cA,cH,cV,cD


def computeImDWTsingleChannel(greyIm, levels, waveletType):
    # Extracts coefficients for a single channel image
    approxIm = greyIm

    cHstack,cVstack,cDstack = [],[],[]

    for i in range(levels):
        coeffs = pywt.dwt2(approxIm, waveletType)
        approxIm, (cH, cV, cD) = coeffs
        cHstack.append(cH)
        cVstack.append(cV)
        cDstack.append(cD)

    return approxIm,cHstack, cVstack, cDstack


def computePercentiles(arrayList,percentile):
    # Extracts the coefficient value corresponding to the X-th percentile (uses every level coefficient at once)
    thrValues = []

    for k in range(arrayList[0].shape[2]):
        vecData = np.reshape(arrayList[0][:, :, 0], (1, -1))
        for i in range(len(arrayList)):
            x = np.reshape(arrayList[i][:,:,k],(1,-1))
            vecData = np.concatenate((vecData,x),1)
        thrValues.append(np.percentile(np.absolute(vecData),percentile))

    return thrValues


def applyThreshold(coeffs,thresholds):
    # Applies the threshold to the coefficients in arrayList. There should be numChannels thresholds (3 for RGB img).
    valCoeffs = []
    for i in range(len(coeffs)): # For every "level" of DWT
        for k in range(coeffs[0].shape[2]): # For every color channel
            B = coeffs[i][:,:,k]
            valCoeffs= np.append(valCoeffs, B[thresholds[k] <= np.absolute(B)])

    return valCoeffs


def sampleCoeffs(coeffs,thrPercentile):
    # Keep only the detail coefficients those geq to thrPercentile
    threshold = computePercentiles(coeffs, thrPercentile)
    subSet = applyThreshold(coeffs, threshold)

    return subSet


def normalizeCoeff(coeffs):
    # use sumFlag to sum all the channels together before normalization (returns greyscale image)
    # use equalFlag to ensure that all coefficients have the same grey background

    coeffs = np.absolute(coeffs[:,:,0]) + np.absolute(coeffs[:,:,1]) + np.absolute(coeffs[:,:,2])

    coeffs = coeffs - coeffs.min()
    coeffs /= coeffs.max()

    coeffs = 1 - coeffs

    return coeffs
