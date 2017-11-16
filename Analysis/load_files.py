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
date = 20171108
sequence = 10
redo_prepare = True
sequence_type = 'thermal_rabi_flop'

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

Raman_pulse_time = []


try:
    with h5py.File('results/' + outfile, 'r') as f:
        f['data']
except KeyError:
    redo_prepare = True
except IOError:
    redo_prepare = True


if redo_prepare:
    print('Preparing {} data...'.format(sequence_type))
    for r, file in matchfiles(folder):
        with h5py.File(os.path.join(r, file), 'r') as h5_file:
            img = h5_file['data']['images' + camera]['Raw'][:]
            
            attrs = h5_file['globals'].attrs
            Raman_pulse_time.append(attrs['Raman_pulse_time'])    
            attrs = h5_file['results/rois_od'].attrs
            attrs
            
        img = np.float64(img)
        od = -np.log(((img[0] < 1) + img[0]) / ((img[1] < 1) + img[1])).T
#        plt.imshow(od, vmin=0, vmax=1)
#        plt.show()
    df = pd.DataFrame()
#    df['od'] = od
#    df['p1'] = p1
#    df['p2'] = p2
    df['Raman_pulse_time'] = Raman_pulse_time
#    df['delta_xyz'] = delta_xyz

    df = df.dropna()
    df = df.sort_values(by='Raman_pulse_time')
#    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')

        


#%%
try:
    with h5py.File(lyse.path, 'r') as h5_file:
        img = h5_file['data']['images' + camera]['Raw'][:]
    img = np.float64(img)

    if camera == 'XZ_Blackfly':
        od = -np.log(((img[1] < 1) + img[1]) / ((img[2] < 1) + img[2])).T
    else:
        od = -np.log(((img[0] < 1) + img[0]) / ((img[1] < 1) + img[1])).T
    xvals = np.arange(od.shape[1]*1.0)
    yvals = np.arange(od.shape[0]*1.0)

    print('Integrated OD is {:.3f} counts'.format(od.sum()))
    
    plt.imshow(od, vmin=0, vmax=2.5)
    # print(od.shape)
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.tight_layout()


except KeyError, IndexError:
    print('No image files found in /data.')