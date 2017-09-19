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
import utils 
from collections import deque
from math import floor
from matplotlib.gridspec import GridSpec

def bin_image(image, n_pixels):
    
    try:
    
        x_shape, y_shape = image.shape    
        image_reshape = image.reshape(int(x_shape / n_pixels), n_pixels, 
                                      int(y_shape / n_pixels), n_pixels)
        binned_image = image_reshape.mean(axis=1)
        binned_image = binned_image.mean(axis=2)
        
    except ValueError:
        print('Image is not divisible by that number of pixels')
        binned_image = image
    
    return binned_image

camera = 'ProEM'
date = 20170831
sequence = 21
redo_prepare = True
fringe_removal = True
crop = True
sequence_type = 'three_state_Ramsey'

positions = [[253, 424], [338, 274], [537, 265]] #starting in mf=0
positions = [[175, 383], [254, 193], [432, 113]] #starting in mf=1, ProEM


n_shots = 5
i = 0
isat = 539.393939394
w_crop = 80

folder = utils.getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

indexed_variable = []
indexed_variable_name = 'free_evolution_time'
roi_sum = []
rois_array = []
od_array = []
probe_list = deque([], n_shots)

try:
    with h5py.File('results/' + outfile, 'r') as f:
        f['data']
except KeyError:
    redo_prepare = True
except IOError:
    redo_prepare = True


if redo_prepare:
    
    i =0
    print('Preparing {} probe list...'.format(sequence_type))
    for r, file in utils.matchfiles(folder):
        i += 1
        if i < n_shots:
            with h5py.File(os.path.join(r, file), 'r') as h5_file:
                try:
                    img = h5_file['data']['images' + camera]['Raw'][:]
                    
                    if camera == 'ProEM':
                        probe = img[0] - img[2]
                        
                    else:
                        probe = img[1] - img[2]
                    probe_list.append(probe)
                    
                except KeyError:
                    
                    print('bad shot')
                
            probe_list.append(probe)
            
    print('Preparing {} data...'.format(sequence_type))
    for r, file in utils.matchfiles(folder):
        with h5py.File(os.path.join(r, file), 'r') as h5_file:
            
            try:
                

            img = h5_file['data']['images' + camera]['Raw'][:]
            
            attrs = h5_file['globals'].attrs
            indexed_variable.append(attrs[indexed_variable_name])      
            
            img = np.float64(img)
            if camera == 'ProEM':
                atoms = img[1] - img[3]
                probe = img[0] - img[2]
                
            else:
                atoms = img[0] - img[2]
                probe = img[1] - img[2]
            
            if fringe_removal:
                probe_list.append(probe)
                opt_ref  = utils.fringeremoval([atoms], probe_list,
                             mask='all', method='svd')
                od = -np.log(((atoms < 1) + atoms) / 
                            ((opt_ref[0] < 1) + opt_ref[0]))
#                od = od + (probe - atoms) / isat
            
            else:
                
                od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
            
#            od = od.T
            od_array.append(od)
            rois = []
            
            for pos in positions:
                
                x,y = pos
                rois.append(od[x - w_crop:x + w_crop, y - w_crop:y + w_crop])

            roi_sum.append(np.sum(rois))
            rois_array.append(rois)

    df = pd.DataFrame()


#    df['p1'] = p1
#    df['p2'] = p2
    df[indexed_variable_name] = indexed_variable
    df['roi_sum'] = roi_sum

    df = df.dropna()
#    df = df.sort_values(by='Raman_pulse_time')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')
    

#%%

n_repetitions = 3
n_points = 120
df['indices'] = df.index / n_repetitions
df  = df.sort_values(by='Raman_pulse_time')
rois_array = np.array(rois_array)
rois_array = rois_array.reshape(n_points, n_repetitions, 3, 
                                2 * w_crop, 2 * w_crop)
rois_mean = np.mean(rois_array, axis=1)
indices = df.indices[::n_repetitions].values
idx = indices[49] 
n_pixels = 4
rois_sorted = []

for idx in indices:
    
    rois_sorted.append(rois_mean[floor(idx), :, :, :])
    
rois_sorted = np.array(rois_sorted)
rois_sorted_sum = rois_sorted.sum(axis=1)

#%%
labels = ['z state', 'x state', 'y state']
tt = df['Raman_pulse_time'][::n_repetitions]
x_pixel = 80
y_pixel = 60

for i in range(3):
    psum = rois_sorted[:,:, x_pixel, y_pixel].sum(axis=1).max()
    psum = 1
    plt.plot(tt, rois_sorted[:,i, x_pixel, y_pixel] / psum, label=labels[i])
plt.legend()        
plt.xlabel('Raman pulse time [us]')
plt.ylabel('Fraction')      
#%%
import scipy.signal as ss
N = len(tt)
dt = tt.values[1] - tt.values[0]
dt = dt * 1e-6
xf = np.linspace(0.0, 1/(2 * dt), int(N / 2))
lspops = np.array([ss.lombscargle(tt.values, rois_sorted[:,i, x_pixel, y_pixel]-rois_sorted[:,i, x_pixel, y_pixel].mean(axis=0) ,
                                  2*np.pi * xf + 1e-15) for i in range(3)])
#%%
plt.plot(xf * 1e-3, lspops[1])
#plt.xlim(0, 200)

#%%    
idx = floor(indices[50])
n_pixels = 2
p0, p1, p2 = rois_mean[idx, :, :, :]
p0 = bin_image(p0, n_pixels)
p1 = bin_image(p1, n_pixels)
p2 = bin_image(p2, n_pixels)
psum = p0 + p1 + p2

gs = GridSpec(1, 4)
fig = plt.figure()

plt.subplot(gs[0])
plt.title('z state')
plt.imshow(p0, interpolation='none', vmin=0, vmax=0.3)

plt.subplot(gs[1])
plt.title('x state')
plt.imshow(p1, interpolation='none', vmin=0, vmax=0.3)

plt.subplot(gs[2])
plt.title('y state')
plt.imshow(p2, interpolation='none', vmin=0, vmax=0.3)

plt.subplot(gs[3])
plt.title('sum')
plt.imshow(psum, interpolation='none', vmin=0, vmax=0.3)


            
#%%
import utils
gs = GridSpec(1, 3)
fig = plt.figure(figsize=(9, 3))      

for x_cut in range(0, 150):
    for i in range(3):
        plt.subplot(gs[i])      
        psum = rois_sorted.sum(axis=1).max()
        psum = 1
        ffts = utils.fft_slice(rois_sorted[0:120,i,:,:]/psum, x_cut, 'x')
        plt.pcolormesh(ffts.T)
        plt.title(labels[i])
    plt.show()
#    plt.yticks(np.linspace(0, 40, 3))
#    plt.ylim([0, 30])
