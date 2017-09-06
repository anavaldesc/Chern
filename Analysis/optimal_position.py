# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:39:08 2017

@author: banano
"""

import sys
sys.path.append('/Users/banano/databandit')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
from fnmatch import fnmatch
#import databandit as db
#import databandit.functions as dbf
import utils 
from collections import deque
from scipy.linalg import pinv, lu, solve

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

camera = 'XY_Flea3'
date = 20170825
sequence = 37
redo_prepare = True
fringe_removal = True
crop = True
sequence_type = 'optimal_position'

positions = [[253, 424], [338, 274], [537, 265]] #starting in mf=0

rois_array = []
i = 0
isat = 539.393939394
w_crop = 60

folder = utils.getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

Raman_pulse_time = []
dummy = []
psum = []
probe_list = deque([], 20)

try:
    with h5py.File('results/' + outfile, 'r') as f:
        f['data']
except KeyError:
    redo_prepare = True
except IOError:
    redo_prepare = True


if redo_prepare:
    print('Preparing {} data...'.format(sequence_type))
    for r, file in utils.matchfiles(folder):
        with h5py.File(os.path.join(r, file), 'r') as h5_file:

            img = h5_file['data']['images' + camera]['Raw'][:]
            
            attrs = h5_file['globals'].attrs
            Raman_pulse_time.append(attrs['Raman_pulse_time']) 
            dummy.append(1.1)
            
            
            img = np.float64(img)
            atoms = img[0] - img[2]
            probe = img[1] - img[2]
            
            if fringe_removal:
                probe_list.append(probe)
                opt_ref  = utils.fringeremoval([atoms], probe_list,
                             mask='all', method='svd')
                od = -np.log(((atoms < 1) + atoms) / 
                            ((opt_ref[0] < 1) + opt_ref[0]))
                od = od + (probe - atoms) / isat
            
            else:
                
                od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
            
            od = od.T
            
            rois = []
            
            for pos in positions:
                
                x,y = pos
                rois.append(od[x - w_crop:x + w_crop, y - w_crop:y + w_crop])
            
            psum.append(np.sum(rois))
            
            rois_array.append(rois)
                
#                od = imutils.rotate_bound(od, theta)
                
#                if crop:
#            
#            od = od[y_crop - wy_crop: y_crop + wy_crop, x_crop - wx_crop: x_crop + wx_crop]
#            
#            try:
#            
#                blob =  dbf.blob_detect(od,show=True,max_sigma=20, min_sigma = 10,
#                             num_sigma=5, threshold=.2)
#                y_blob, x_blob = blob[0][0:2] 
#                x_blob += x_crop - wx_crop
#                y_blob += y_crop - wy_crop
#                print[blob[0][0:2]]
#            
#          
#            except Exception as e:
#                
#                y_blob, x_blob = [np.nan, np.nan]
##                print (y_blob, x_blob)
##                print (e)
#    
#        else:
#            
#            try:
#            
#                blob =  dbf.blob_detect(od,show=True,max_sigma=30, min_sigma = 25,
#                             num_sigma=5, threshold=.2)
#                y_blob, x_blob = blob[0][0:2] 
#                print(blob)
#            
##                print(blob)
#            except TypeError:
#                
#                 y_blob, x_blob = [np.nan, np.nan]
                
            
    
#        x_val.append(x_blob)
#        y_val.append(y_blob)
#        plt.imshow(od, vmin=0, vmax=1)
#        plt.show()
    df = pd.DataFrame()
#    df['x_val'] = x_val
#    df['y_val'] = y_val

#    df['p1'] = p1
#    df['p2'] = p2
    df['Raman_pulse_time'] = Raman_pulse_time
    df['dummy'] = dummy
    df['psum'] = psum

    df = df.dropna()
#    df = df.sort_values(by='Raman_pulse_time')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')
    

#%%

from matplotlib.gridspec import GridSpec
df['indices'] = df.index / 3
df  = df.sort_values(by='Raman_pulse_time')
n_repetitions = 3
rois_array = np.array(rois_array)
rois_array = rois_array.reshape(50, n_repetitions, 3, 120, 120)
rois_mean = np.mean(rois_array, axis=1)
indices = df.indices[::3].values
idx = indices[49] 
n_pixels = 4
rois_sorted = []
for idx in indices:
    
    rois_sorted.append(rois_mean[idx, :, :, :])

rois_sorted = np.array(rois_sorted)
rois_sorted_sum = rois_sorted.sum(axis=1)

for i in range(3):
    plt.plot(rois_sorted[:,i, 15, 29])        
      

#%%    
idx = indices[30] 
n_pixels = 1
p0, p1, p2 = rois_mean[idx, :, :, :]
p0 = bin_image(p0, n_pixels)
p1 = bin_image(p1, n_pixels)
p2 = bin_image(p2, n_pixels)
psum = p0*0 + p1 + p2

gs = GridSpec(1, 4)
fig = plt.figure()

plt.subplot(gs[0])
plt.title('z state')
plt.imshow(p0, interpolation='none', vmin=0, vmax=0.4)

plt.subplot(gs[1])
plt.title('x state')
plt.imshow(p1, interpolation='none', vmin=0, vmax=0.4)

plt.subplot(gs[2])
plt.title('y state')
plt.imshow(p2, interpolation='none', vmin=0, vmax=0.4)

plt.subplot(gs[3])
plt.title('sum')
plt.imshow(psum, interpolation='none', vmin=0, vmax=0.4)

#
#for i in range(50):
#    for j in range(3):
#    
#    
#        mean_roi = np.mean(rois_array)


#for j in range(3):
#    for i in range(9, 12):
#    
#        p0 = (rois_array[9, 0, :, :] + rois_array[10, 0, :, :] + rois_array[11, 0, :, :]) / 3
#%%

def fft_slice(images_stack, pos, direction):
    
    psd_arr = []
    n, x_dim, y_dim = images_stack.shape
#    print (n)
    
    if direction == 'x':
        for i in range(0, x_dim):
            image_slice = images_stack[:, i, pos] - images_stack[:, i, pos].mean()
            fft = np.fft.fftshift(np.fft.fft(image_slice))
            psd = np.abs(fft[n/2::])**2
            psd_arr.append(psd)
    
    elif direction == 'y':
        for i in range(0, y_dim):
            image_slice = images_stack[:, pos, i] - images_stack[:, pos, i].mean()
            fft = np.fft.fftshift(np.fft.fft(image_slice))
            psd = np.abs(fft[n/2::])**2
            psd_arr.append(psd)
    
    psd_arr = np.array(psd_arr)
    return psd_arr
        
            
#%%

gs = GridSpec(1, 3)
fig = plt.figure()      

for i in range(3):
    plt.subplot(gs[i])      
    ffts = fft_slice(rois_sorted[:,i,:,:] / rois_sorted_sum, 15, 'x')
    plt.imshow(ffts.T, aspect='auto', interpolation='none')
    plt.yticks(np.linspace(0, 20, 3))
    plt.ylim([0, 20])
