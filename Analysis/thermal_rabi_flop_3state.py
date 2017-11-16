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
date = 20170713
sequence = 29
redo_prepare = True
sequence_type = 'thermal_rabi_flop'
x = [78, 329, 534]
y = [281, 344, 247] #start in z 
wx = 75
wy = 75
pixels = np.zeros((2 * wx, 2 * wy, 3))
pixels_array = []

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

Raman_pulse_time = []
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
                Raman_pulse_time.append(attrs['Raman_pulse_time'])            
                
                img = np.float64(img)
                atoms = img[0] - img[2]
                probe = img[1] - img[2]
                
                pixels = np.zeros((2 * wx, 2 * wy, 3))

                for i in range(3):
                    
                    od_crop = od[x[i]-wx:x[i]+wx, y[i]-wy:y[i]+wy]
#                    plt.imshow(od_crop, vmin=0, vmax=1)
#                    plt.show()
                    pixels[:,:,i] = od_crop
                    
                pixels_array.append(pixels)

                
        df = pd.DataFrame()
        df['Raman_pulse_time'] = Raman_pulse_time
        
#        for i in range(3):
#            pixels[:,:,i] /= pixels.sum(axis=2)
#        for i in range(2 * wx):
#            for j in range(2 * wy):
#                df['p0_{},{}'.format(i,j)] = pixels[i, j, 0]
#        for i in range(wx):
#            df['frac{}'.format(i)] = fracs[:,i]
    
    #            for i in range(50):
#        df['frac'] = fraction[:,1:1+2].mean()
    #            print(fraction[:,1:1+4].mean())
    #                print(fr action[:,i:i+4].mean())
    #    df['delta_xyz'] = delta_xyz
    
        df = df.dropna()
        df = df.sort_values(by='Raman_pulse_time')
    #    df.to_hdf('results/' + outfile, 'data', mode='w')
    else:
        df = pd.read_hdf('results/' + outfile, 'data')

except Exception as e:
#    print('Fix your analysis')
    print(e)        


#%%
df = pd.DataFrame()
df['Raman_pulse_time'] = Raman_pulse_time
pixels_array = np.array(pixels_array)

for i in range(3):
    pixels_array[:, :,:,i] /= pixels_array.sum(axis=3)
for i in range(2 * wx): 
    for j in range(2 * wy):
        for k in range(3):
            df['p{}_{},{}'.format(k,i,j)] = pixels_array[:, i, j, k]
#        for i in range(wx):
#            df['frac{}'.format(i)] = fracs[:,i]

#            for i in range(50):
#        df['frac'] = fraction[:,1:1+2].mean()
#            print(fraction[:,1:1+4].mean())
#                print(fr action[:,i:i+4].mean())
#    df['delta_xyz'] = delta_xyz

df = df.dropna()
df = df.sort_values(by='Raman_pulse_time')
#%%
for i in range(3):
        plt.plot(df['Raman_pulse_time'], df['p{}_75,75'.format(i)],'.--')
plt.ylim([0,1])
plt.xlabel('Raman pulse time [us]')
plt.ylabel('Fraction')
#%%


for i in range(2 * wx):
    sorted_fractions.append(df['frac{}'.format(i)])
    plt.plot(df['Raman_pulse_time'], df['frac{}'.format(i)])
    plt.ylim([0,1])
    plt.xlabel('Raman pulse time [us]')
    plt.ylabel('Fraction')
plt.show()


#%%
p = 2
psd_vec = []

for p in range(3):
    for i in range(2 * wy):
        row = pixels_array[:, i, wy, p]
        row_ft = np.fft.fftshift( np.fft.fft(row - row.mean()))
        N = len(row_ft)
        psd = np.abs(row_ft[N/2::])**2
        #    plt.plot(psd)
        psd_vec.append(psd/psd.max())

psd_vec = np.array(psd_vec)
psd_vec = psd_vec.reshape((3, 2 * wx, N /2))
psd_mean = np.mean(psd_vec, axis=0)
#%%
psd /= psd.max()
tvec = df.as_matrix().T[0]
d = tvec[1] - tvec[0]
d = d*1e-6
freqs = np.fft.fftfreq(N, d)[0:N/2]*1e-3
plt.pcolormesh(psd_mean.T, vmin=0, vmax=1, cmap='YlGn')
plt.xlabel('pixel')
plt.ylabel('Frequency [kHz]')
plt.yticks(range(0, N/2, N/10), ['%.1f'%freqs[::int(N/10)][i] for i in range(0,5)])
plt.axis('Tight')


