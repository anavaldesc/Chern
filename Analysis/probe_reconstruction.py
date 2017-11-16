#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:05:36 2017

@author: banano
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:39:08 2017

@author: banano
"""
import sys
sys.path.append('/Users/banano/Documents/UMD/Research/Rashba/Chern/Utils/')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
from fnmatch import fnmatch
from tqdm import tqdm
import utils
from image_reconstruction.cpu_reconstructor import CPUReconstructor as ImageReconstructor

def reconstruct_probes(mask, raw_probes, raw_atoms):
   reconstructor = ImageReconstructor()
   reconstructor.add_ref_images(raw_probes)
   reconstructed_probes = []
   for atoms in tqdm(raw_atoms):
       reconstructed_probes.append(reconstructor.reconstruct(image=atoms, mask=mask)[0])
   del reconstructor
   return reconstructed_probes

camera = 'XY_Flea3'
date = 20170906
sequence = 59
redo_prepare = True
sequence_type = 'thermal_rabi_flop'
x = [78, 329, 534]  #start in z 
y = [281, 344, 247] #start in z 

y = [191, 298, 518]# start in x
x = [389, 173, 96] # start in x
wx = 80
wy = 80
n_reps = 4
n_shots = 90
isat = 530

pixels = np.zeros((2 * wx, 2 * wy, 3))
pixels_array = []

folder = utils.getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

Raman_pulse_time = []
fracs = []
probe_list = []
atoms_list = []
dark_list = []


try:
    with h5py.File('results/' + outfile, 'r') as f:
        f['data']
except KeyError:
    redo_prepare = True
except IOError:
    redo_prepare = True


try:
    if redo_prepare:
        print('Preparing {} image lists...\n'.format(sequence_type))
        for r, file in utils.matchfiles(folder):
            with h5py.File(os.path.join(r, file), 'r') as h5_file:
                img = h5_file['data']['images' + camera]['Raw'][:]
                
                attrs = h5_file['globals'].attrs
                Raman_pulse_time.append(attrs['Raman_pulse_time'])            
                
                img = np.float64(img)
                atoms = img[0].T
                probe = img[1].T
                dark = img[2].T
                atoms_list.append(atoms)
                probe_list.append(probe)
                dark_list.append(dark_list)

                
        df = pd.DataFrame()
        df['Raman_pulse_time'] = Raman_pulse_time
    
#        df = df.dropna()
#        df = df.sort_values(by='Raman_pulse_time')
    #    df.to_hdf('results/' + outfile, 'data', mode='w')
    else:
        df = pd.read_hdf('results/' + outfile, 'data')

except Exception as e:
#    print('Fix your analysis')
    print(e)        
#%%
print('Reconstructing probes...')
roi_mask = np.ones(atoms_list[0].shape)
w_mask = 110
for i in range(3):
    
    roi_mask[y[i] - w_mask: y[i] + w_mask, x[i] - w_mask:x[i] + w_mask] = 0


rprobes = reconstruct_probes(roi_mask, probe_list, atoms_list)
#%%
from math import floor
df['avg_index'] = [floor(df.index.values[i] / n_reps) for i in range(len(df))] 
df['repetition_index'] = df.index.values % n_reps
df['sorting_index'] = df.index.values
#for 
idx = 14
od_r = -np.log(atoms_list[idx]/ rprobes[idx])
od = -np.log(atoms_list[idx]/ probe_list[idx])

plt.imshow(od_r, cmap='jet', vmin=-0.01, vmax=0.4)

#sorting_indices = np.zeros([n_reps, n_shots])

sorting_indices = df.sort_values(by='Raman_pulse_time')[df.repetition_index == 0].index.values 
    
#%%
print('Calculating od with reconstructed probes...')
od_list = []


for j in range(4):
    for idx in sorting_indices:
     
        atoms = atoms_list[idx + j]
        rprobe = rprobes[idx + j]

#for atoms, rprobe in zip(atoms_list, rprobes):
    
        od = -np.log(((atoms < 1) + atoms) / ((rprobe < 1) + rprobe))
        
        if isat > 0:
            
            od += (rprobe - atoms) / isat
    #    plt.imshow(od)
    #    plt.show()
        od_list.append(od)
    
#%%

#for od in od_list:
#    plt.imshow(od)
#    plt.show()
#%%
wx = 80
wy = 80
print('Making stacks of rois...')
#pixels = np.zeros((2 * wx, 2 * wy, 3))
rois_stack = []
integrated_od = []
for banana in od_list:
    pixels = np.zeros((2 * wx, 2 * wy, 3))
    for i in range(3):
        
        od_crop = banana[y[i]-wy:y[i]+wy, x[i]-wx:x[i]+wx]
        pixels[:,:,i] = od_crop
#    plt.imshow(od)
#    plt.show()
    integrated_od.append(pixels.sum())        
    rois_stack.append(pixels)
    
rois_stack = np.array(rois_stack)

df['integrated_od'] = integrated_od
#%%
stack_reshape = np.reshape(rois_stack, (n_reps, n_shots, 2 * wy, 2 * wx, 3))


#plot one pixel
y_coord = int(wy / 2)
x_coord = int(wx / 2)
#for i in range(90):
#    plt.imshow(stack_reshape.mean(axis=0)[i, :, :, 1], vmin=0, vmax=0.5)
#    plt.show()

psum = np.array(stack_reshape.mean(axis=0).sum(axis=3))    
#for i in range(3):
#    pops = stack_reshape.mean(axis=0)[:, y_coord, x_coord, i]
#    pops /= psum[:, y_coord, x_coord]
#    plt.plot(pops)
#    plt.show()
    
print('Calculating ffts...')

psd_array = np.zeros((int(n_shots / 2), 2 * wy, 2 * wx, 3))
mean_roi_stack = stack_reshape.mean(axis=0)
normalize_psd = False

for i in range(2 * wy):
    for j in range(2 * wx):
        for k in range(3):
            
            time_trace = mean_roi_stack[:, i, j, k]
            time_trace -= time_trace.mean()
            fft = np.fft.fftshift(np.fft.fft(time_trace))
            psd = np.abs(fft[int(n_shots / 2)::])**2
            if normalize_psd:
                psd = psd / psd.max()
            psd_array[:, i, j, k] = psd
            
#%%
for i in range(2 * wy):       
    plt.pcolormesh(psd_array[:, :, i, 0])
    
    plt.show()
    
#%%
    
from mayavi import mlab    
s = psd_array.sum(axis=3)[0:20]
mlab.pipeline.volume(mlab.pipeline.scalar_field(s), vmin=0, vmax=20)
mlab.show()