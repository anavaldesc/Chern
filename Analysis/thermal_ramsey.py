# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:39:08 2017

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
from fnmatch import fnmatch
#import databandit as db

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



camera = 'XY_Flea3'
date = 20170703
sequence = 61
redo_prepare = True
sequence_type = 'thermal_rabi_flop'
x0 = [233, 393]
x1 = [319, 174] # center location of atomic states involved in Rabi flop
wx = 100
wy = 50
od0vec = []
od1vec = []
fraction = []

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

free_evolution_time = []
fracs = []


try:
    with h5py.File('results/' + outfile, 'r') as f:
        f['data']
except KeyError:
    redo_prepare = True
except IOError:
    redo_prepare = True


try:
    if redo_prepare:
        print('Preparing {} data...'.format(sequence_type))
        for r, file in matchfiles(folder):
            with h5py.File(os.path.join(r, file), 'r') as h5_file:
                img = h5_file['data']['images' + camera]['Raw'][:]
                
                attrs = h5_file['globals'].attrs
                free_evolution_time.append(attrs['free_evolution_time'])            
                
                img = np.float64(img)
                atoms = img[0] - img[2]
                probe = img[1] - img[2]
                od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
                od0 = od[x0[0]-wx*1:x0[0]+wx, x0[1]-wy:x0[1]+wy].T
                od1 = od[x1[0]-wx*1:x1[0]+wx, x1[1]-wy:x1[1]+wy].T
                od0vec.append(od0)
                od1vec.append(od1)
#                fraction.append(od0 / (od0 + od1))
    
                
#                for i in range(100):
#                    frac_row.append(fraction[:,i:i+2].mean())
#                fracs.append(frac_row)
    #        fracs = np.array(fracs)
                
#                plt.imshow(od.T, vmin=0, vmax=1)
#                plt.show()
               
        print('banana')
        df = pd.DataFrame()
        df['free_evolution_time'] = free_evolution_time
        od1vec = np.array(od1vec)
        od0vec = np.array(od0vec)
        
        for i in range(len(od1vec)):
            mean1 = od1vec[i].mean(axis=0)
            mean0 = od0vec[i].mean(axis=0)
            frac = mean0 / (mean0 + mean1)
            fraction.append(frac)
        fraction = np.array(fraction)
        for i in range(len(fraction[0])):
            df['pixel{}'.format(i)] = fraction[:,i]
    
    #            for i in range(50):
#        df['frac'] = fraction[:,1:1+2].mean()
    #            print(fraction[:,1:1+4].mean())
    #                print(fr action[:,i:i+4].mean())
    #    df['delta_xyz'] = delta_xyz
    
        df = df.dropna()
        df = df.sort_values(by='free_evolution_time')
        df.to_hdf('results/' + outfile, 'data', mode='w')
    else:
        df = pd.read_hdf('results/' + outfile, 'data')

except KeyError:
    print('Empty data file')        


#%%
sorted_fractions = df.as_matrix().T[1::].T
#for i in range(len(sorted_fractions)):
#    sorted_fractions[i] *= 1/sorted_fractions[i].max()
plt.pcolormesh(sorted_fractions, cmap='viridis_r', vmin=0, vmax=1)
#           vmin=0, vmax=1)
plt.xlabel('pixel')
plt.ylabel('free evolution time [us]')
plt.yticks(range(0, 30, 5), ['%.1f'%df.as_matrix().T[0][::5][i] for i in range(0,6)])
plt.axis('Tight')
plt.show()

#fft = np.fft.fft2(sorted_fractions-np.mean(sorted_fractions))
#psd = np.abs(fft)**2
#psd /= psd.max()
#plt.imshow(psd, vmin=0, vmax=0.1)
#%%
psd_vec = []
for i in range(len(sorted_fractions)):
    frac_ft = np.fft.fftshift( np.fft.fft(sorted_fractions[i]-sorted_fractions[i].mean()))
    N = len(frac_ft)
    psd = np.abs(frac_ft[N/2::])**2
#    plt.plot(psd)
    psd_vec.append(psd)

psd = np.array(psd_vec)
psd /= psd.max()
N = len(sorted_fractions[0])
freqs = np.fft.fftfreq(N, d=1)[0:N/2]
plt.pcolormesh(psd, vmin=0, vmax=1, cmap='YlGn')
plt.axis('Tight')
plt.ylabel('free evolution time [us]')
plt.yticks(range(0, 30, 5), ['%.1f'%df.as_matrix().T[0][::5][i] for i in range(0,6)])
plt.xlabel('Spatial frequency [1/pixel]')
plt.xticks(range(0, N/2, N/10), ['%.1f'%freqs[::int(N/10)][i] for i in range(0,5)])
plt.show()

#%%

psd_vec = []
for i in range(len(sorted_fractions[0])):
    frac_ft = np.fft.fftshift( np.fft.fft(sorted_fractions[:,i]-sorted_fractions[:,i].mean()))
    N = len(frac_ft)
    psd = np.abs(frac_ft[N/2::])**2
#    plt.plot(psd)
    psd_vec.append(psd)

psd = np.array(psd_vec)
psd /= psd.max()
N = len(sorted_fractions[:,0])
fet = df.as_matrix().T[0]
d = fet[1] - fet[0]
d = d*1e-6
freqs = np.fft.fftfreq(N, d)[0:N/2]*1e-3
plt.pcolormesh(psd.T, vmin=0, vmax=1, cmap='YlGn')
plt.xlabel('pixel')
plt.ylabel('Frequency [kHz]')
plt.yticks(range(0, N/2, N/10), ['%.1f'%freqs[::int(N/10)][i] for i in range(0,5)])
plt.axis('Tight')
plt.show()
#for i in range(50):
#    plt.plot(df['free_evolution_time'], df['frac{}'.format(i)])
#    plt.ylim([0,1])
#    plt.xlabel('Raman pulse time [us]')
#    plt.ylabel('Fraction')
#
#%%
import imutils
idx = 16 # 14 isa good example also 16
thetas = np.linspace(-5, 8, 50)
plt.imshow(od0vec[idx]/ (od0vec[idx]+od1vec[idx]), vmin=0, vmax=1)
plt.show()
contrast = []
reference = []
#plt.show()
#plt.imshow(img, vmin=0, vmax=0.5)
for theta in thetas:
    img = imutils.rotate_bound(od0vec[idx]/ (od0vec[idx]+od1vec[idx]), theta)
    masked_img = np.ma.masked_array(img, img==0)
    flat_array_masked = np.mean(masked_img, axis=0)
    flat_array = np.mean(img, axis=0)
    flat_no_rotation = np.mean(od0vec[idx]/ (od0vec[idx]+od1vec[idx]), axis=0)
#    plt.show()
#    plt.plot(flat_array_masked, 'k')
    #plt.plot(flat_array, 'r')
#    plt.plot(flat_no_rotation, 'b')
    contrast.append(flat_array_masked[30:-30].max() - flat_array_masked[30:-30].min())
    reference.append(flat_no_rotation[30:-30].max()- flat_no_rotation[30:-30].min())
    
plt.plot(thetas, contrast, 'k', label='rotated image')
plt.plot(thetas, reference, 'b', label='reference image')
plt.xlabel("rotation angle")
plt.ylabel("contrast")
plt.legend()
#%%
plt.plot(flat_array_masked, 'k', label='8 degree rotation')
#plt.plot(flat_array, 'r')
plt.plot(flat_no_rotation, 'b', label='no rotation')
#plt.legend()
plt.axis('Tight')
##%%
#dfod = pd.DataFrame()
#dfod['free_evolution_time'] = free_evolution_time
#
#for i in range(len(od0vec)):
#    vertical_mean = od0vec[i].mean(axis=0)
#    plt.plot(od0vec[i].mean(axis=0))
#
##%%
#i = 0
#od1vec = np.array(od1vec)
#od0vec = np.array(od0vec)
#mean1 = od1vec[i].mean(axis=0)
#mean0 = od0vec[i].mean(axis=
#plt.plot(od0vec[i].mean(axis=0))
#plt.plot(od1vec[i].mean(axis=0))
#plt.plot(mean0/(mean0+mean1))
#plt.xlabel('pixel number')
#plt.ylabel('optical depth')
#
##%%
#fraction = []
#df = pd.DataFrame()8 fegre
#df['free_evolution_time'] = free_evolution_time
#od1vec = np.array(od1vec)
#od0vec = np.array(od0vec)
#
#for i in range(len(od1vec)):
#    mean1 = od1vec[i].mean(axis=0)
#    mean0 = od0vec[i].mean(axis=0)
#    frac = mean0 / (mean0 + mean1)
#    fraction.append(frac)
#fraction = np.array(fraction)
##%%
#for i in range(200):
#    df['pixel{}'.format(i)] = fraction[:,i]