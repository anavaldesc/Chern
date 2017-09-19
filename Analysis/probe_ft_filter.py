#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:31:16 2017

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


camera = 'XY_Flea3'
date = 20170726
sequence = 56
redo_prepare = True
sequence_type = 'x_0kr_position'


folder = utils.getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

#Raman_pulse_time = []
atoms = []
probe = []
fts = []


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
            
            try:
    
                img = h5_file['data']['images' + camera]['Raw'][:]
                attrs = h5_file['globals'].attrs
                
            except KeyError:
                
                print('There are no {} images in this file'.format(camera))
    #            Raman_pulse_time.append(attrs['Raman_pulse_time']) 
            
        
        img = np.float64(img)
        if camera == 'ProEM':
            
            a = img[1] - img[3]
            p = img[0] - img[2]
            ft = np.fft.fftshift(np.fft.fft2(p - p.mean()))
            
        
        elif camera == 'XY_Flea3':
            isat = 539.393939394
            a = img[0] - img[2]
            p = img[1] - img[2]
            ft = np.fft.fftshift(np.fft.fft2(p - p.mean()))
        
        atoms.append(a)
        probe.append(p)
        fts.append(ft)



#%%

#od = -np.log(((a < 1) + a) / ((p < 1) + p)) + (p - a) / isat
#plt.imshow(od[300:400, 200:300])
#cropped_od = od[368-50:368+50, 274-50:274+50]
#blob1 =  dbf.blob_detect(od,show=True,max_sigma=30, min_sigma = 25,
#                             num_sigma=5, threshold=.2)
#blob2 =  dbf.blob_detect(cropped_od,show=True,max_sigma=30, min_sigma = 25,
#                             num_sigma=5, threshold=.2)
##plt.show()
#plt.pcolormesh(od)mean(df['x_val'])
#ymean = np.nanmean(df['y_val'])
#