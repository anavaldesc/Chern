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
import databandit as db


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
sequence = 22
redo_prepare = True
sequence_type = 'bec_rabi_flop_3state'
#x = [78, 329, 534]
#y = [281, 344, 247] #start in z
#x = [80, 331, 530]
#y = [178, 241, 151] #start in x
x = [124, 375, 579]
y = [231, 292, 202] #start in y
wx = 120
wy = 30
w = 30
ods = []
iod = []

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
                od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
                iod.append(np.ma.masked_invalid(od).sum())
                
                grid = db.functions.imagegrid([0,0], [488, 648], atoms)
                
                od_row = []
                for i in range(3):
                    
                   od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
                   p = [x[i], w, y[i], w]
                   od = od * db.qgasfunctions.puck(grid, p)   
                   od_row.append(np.ma.masked_invalid(od).sum())
#                   plt.imshow(od)
#                   plt.show()
                

                od_row = np.array(od_row)                 
                ods.append(od_row / od_row.sum())
                   
##%%
#                od0 = od[x0[0]-wx:x0[0]+wx, x0[1]-wy:x0[1]+wy].T
#                od1 = od[x1[0]-wx:x1[0]+wx, x1[1]-wy:x1[1]+wy].T
#                fraction = od0 / (od0 + od1)
#                frac_row = []
#                
#                for i in range(wx):
#                    frac_row.append(fraction[:,i:i+2].mean())
#                fracs.append(frac_row)
#                print(fracs)
    #        fracs = np.array(fracs)
                
#                plt.imshow(od.T, vmin=0, vmax=1)
#                plt.show()
                
        df = pd.DataFrame()
        df['Raman_pulse_time'] = Raman_pulse_time
        df['integratedOD'] = iod
        fracs = np.array(fracs)
        ods = np.array(ods)
        for i in range(3):
            df['p{}'.format(2-i)] = ods[:,i]
    
    #            for i in range(50):
#        df['frac'] = fraction[:,1:1+2].mean()
    #            print(fraction[:,1:1+4].mean())
    #                print(fr action[:,i:i+4].mean())
    #    df['delta_xyz'] = delta_xyz
    
        df = df.dropna()
#        df = df.sort_values(by='Raman_pulse_time')
#        df = df[ df.integratedOD > df.integratedOD.mean() * 1 ]
    #    df.to_hdf('results/' + outfile, 'data', mode='w')
    else:
        df = pd.read_hdf('results/' + outfile, 'data')

except KeyError:
    print('Empty data file')        


#%%
for i in range(3):
    
    plt.plot(df['Raman_pulse_time'], df['p{}'.format(i)], 'o')
    plt.xlabel('Raman pulse time')
    plt.ylabel('Fraction')

#%%
sorted_fractions = []
for i in range(wx):
    sorted_fractions.append(df['frac{}'.format(i)])
    plt.plot(df['Raman_pulse_time'], df['frac{}'.format(i)])
    plt.ylim([0,1])
    plt.xlabel('Raman pulse time [us]')
    plt.ylabel('Fraction')
plt.show()

sorted_fractions = np.array(sorted_fractions)
psd_vec = []
for i in range(len(sorted_fractions[:,0])):
    frac_ft = np.fft.fftshift( np.fft.fft(sorted_fractions[i]-sorted_fractions[i].mean()))
    N = len(frac_ft)
    psd = np.abs(frac_ft[N/2::])**2
#    plt.plot(psd)
    psd_vec.append(psd/psd.max())

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


