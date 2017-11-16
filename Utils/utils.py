# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:23:49 2017

@author: banano
"""

#from collections import deque
import sys
sys.path.append('/Users/banano/Documents/UMD/Research/Rashba/Chern/Utils/')

from scipy.linalg import pinv, lu, solve
import numpy as np
import pandas as pd
import os
from fnmatch import fnmatch
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from image_reconstruction.cpu_reconstructor import CPUReconstructor as ImageReconstructor
from tqdm import tqdm


def matchfiles(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if fnmatch(file, '*.h5'):
                yield root, file
        break  # break after listing top level dir


def getfolder(date, sequence):
    date = pd.to_datetime(str(date))
    folder = 'data/' + date.strftime('%Y/%m/%d') + '/{:04d}'.format(sequence)
    return folder

def fringeremoval(img_list, ref_list, mask='all', method='svd'):

    nimgs = len(img_list)
    nimgsR = len(ref_list)
    xdim = img_list[0].shape[0]
    ydim = img_list[0].shape[1]

    if mask == 'all':
        bgmask = np.ones([ydim, xdim])
        # around 2% OD reduction with no mask
    else:
        bgmask = mask

    k = (bgmask == 1).flatten(1)

    # needs to be >float32 since float16 doesn't work with linalg

    R = np.dstack(ref_list).reshape((xdim*ydim, nimgsR)).astype(np.float64)
    A = np.dstack(img_list).reshape((xdim*ydim, nimgs)).astype(np.float64)

    # Timings: for 50 ref images lasso is twice as slow
    # lasso 1.00
    # svd 0.54
    # lu 0.54

    optref_list = []

    for j in range(A.shape[1]):

        if method == 'svd':
            b = R[k, :].T.dot(A[k, j])
            Binv = pinv(R[k, :].T.dot(R[k, :]))  # svd through pinv
            c = Binv.dot(b)
            # can also try linalg.svd()

        elif method == 'lu':
            b = R[k, :].T.dot(A[k, j])
            p, L, U = lu(R[k, :].T.dot(R[k, :]))
            c = solve(U, solve(L, p.T.dot(b)))

        elif method == 'lasso':
            lasso = Lasso(alpha=0.01)
            lasso.fit(R, A)
            c = lasso.coef_

        else:
            raise Exception('Invalid method.')

        optref_list.append(np.reshape(R.dot(c), (xdim, ydim)))

    return optref_list

def blob_detect(img, show=False, **kwargs):

    if kwargs is None:
        kwargs = {'min_sigma': 5,
                  'max_sigma': 20,
                  'num_sigma': 15,
                  'threshold': 0.05}

    # image = plt.imread('blobs2.png')
    # image_gray = rgb2gray(image)

    blobs = blob_log(img, **kwargs)
    # Compute radii in the 3rd column.
    try:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    except IndexError:
        blobs = None

    if show:

        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
            ax.add_patch(c)

        plt.show()

    return blobs

def bin_image(image, n_pixels):
    
    try:
    
        x_shape, y_shape = image.shape    
        image_reshape = image.reshape(x_shape / n_pixels, n_pixels, 
                                      y_shape / n_pixels, n_pixels)
        binned_image = image_reshape.mean(axis=1)
        binned_image = binned_image.mean(axis=2)
        
    except ValueError:
        print('Image is not divisible by that number of pixels')
        binned_image = image
    
    return binned_image


def fft_slice(images_stack, pos, direction, method='fft'):
    
    psd_arr = []
    n, x_dim, y_dim = images_stack.shape
#    print (n)
    
    if direction == 'x':
        for i in range(0, x_dim):
            image_slice = images_stack[:, i, pos] - images_stack[:, i, pos].mean()
            fft = np.fft.fftshift(np.fft.fft(image_slice))
            psd = np.abs(fft[int(n/2)::])**2
            psd = psd / psd.max()
            psd_arr.append(psd)
    
    elif direction == 'y':
        for i in range(0, y_dim):
            image_slice = images_stack[:, pos, i] - images_stack[:, pos, i].mean()
            fft = np.fft.fftshift(np.fft.fft(image_slice))
            psd = np.abs(fft[int(n/2)::])**2
            psd = psd / psd.max()
            psd_arr.append(psd)
    
    psd_arr = np.array(psd_arr)
    return psd_arr

def reconstruct_probes(mask, raw_probes, raw_atoms):
    reconstructor = ImageReconstructor()
    reconstructor.add_ref_images(raw_probes)
    reconstructed_probes = []
    for atoms in tqdm(raw_atoms):
        reconstructed_probes.append(reconstructor.reconstruct(image=atoms, mask=mask)[0])
    del reconstructor
    return reconstructed_probes