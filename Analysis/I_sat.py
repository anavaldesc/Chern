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
import databandit.functions as dbf

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
date = 20170823
sequence = 173
redo_prepare = False
sequence_type = 'I_sat'
crop = True
find_center = False
x_crop = 370#243 #[79, 328, 535], [80, 332, 178], [124, 375, 579]
y_crop = 279#323 # [279, 344, 252], [178, 244, 151], [231, 292, 202]
w_crop = 10
i = 0

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

#Raman_pulse_time = []
x_val = []
y_val = []

probe_vals =[]
atoms_vals = []

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
#            Raman_pulse_time.append(attrs['Raman_pulse_time']) 
            
        
        img = np.float64(img)
        atoms = img[0] - img[2]
        probe = img[1] - img[2]
        od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
        
        if crop:
            
            od = od[x_crop - w_crop: x_crop + w_crop, 
                    y_crop - w_crop: y_crop + w_crop]
            atoms = atoms[x_crop - w_crop: x_crop + w_crop, 
                          y_crop - w_crop: y_crop + w_crop]
            probe = probe[x_crop - w_crop: x_crop + w_crop, 
                          y_crop - w_crop: y_crop + w_crop]
            atoms_vals.append(np.nanmean(atoms))
            probe_vals.append(np.nanmean(probe))
            
        else:
            
            atoms_vals.append(np.nanmean(atoms))
            probe_vals.append(np.nanmean(probe))
            
        if find_center:
            
            try:
            
                blob =  dbf.blob_detect(od,show=False,max_sigma=120, min_sigma = 100,
                             num_sigma=5, threshold=.2)
                y_blob, x_blob = blob[0][0:2] + [- w_crop + y_crop, -w_crop + x_crop]
            
          
            except Exception as e:
                
                y_blob, x_blob = [np.nan, np.nan]
                print (y_blob, x_blob)
                print (e)
    

            x_val.append(x_blob)
            y_val.append(y_blob)
#        
        else:
            
            x_val.append(0)
            y_val.append(0)
            
    df = pd.DataFrame()
    df['x_val'] = x_val
    df['y_val'] = y_val
    df['atoms'] = atoms_vals
    df['probe'] = probe_vals

#    df['p1'] = p1
#    df['p2'] = p2
#    df['Raman_pulse_time'] = Raman_pulse_time
#    df['delta_xyz'] = delta_xyz

    df = df.dropna()
    df = df.sort_values(by='probe')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')
    

def number(atoms, probe, isat):
    
    n = - np.log(atoms / probe) + (probe - atoms) / isat
    return n
    
isat = 800
isat_vals = np.linspace(400, 700, 100)
std_vals = []
for isat in isat_vals:
    
    num  = []
    for a, p in zip(df['atoms'], df['probe']):
        n = number(a, p, isat)
        num.append(n)
    std_vals.append(np.std(num))

min_idx = np.argmin(std_vals)
plt.plot(isat_vals[min_idx], std_vals[min_idx], '*', isat_vals, std_vals, '-')
#plt.plot(isat_vals, std_vals, '-')
print(isat_vals[min_idx])
