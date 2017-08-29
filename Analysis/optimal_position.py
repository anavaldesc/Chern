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
import utils 
from collections import deque
from scipy.linalg import pinv, lu, solve


camera = 'XY_Flea3'
date = 20170825
sequence = 37
redo_prepare = True
fringe_removal = True
crop = True
sequence_type = 'optimal_position'

positions = [[252, 427], [339, 276], [532, 267]]
rois_array = []
i = 0
isat = 539.393939394
w_crop = 60

folder = utils.getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

#Raman_pulse_time = []
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
#            Raman_pulse_time.append(attrs['Raman_pulse_time']) 
            
            
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
        plt.imshow(od, vmin=0, vmax=1)
        plt.show()
    df = pd.DataFrame()
#    df['x_val'] = x_val
#    df['y_val'] = y_val

#    df['p1'] = p1
#    df['p2'] = p2
#    df['Raman_pulse_time'] = Raman_pulse_time
#    df['delta_xyz'] = delta_xyz

    df = df.dropna()
#    df = df.sort_values(by='Raman_pulse_time')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')
    

#%%

rois = np.array(rois)
p0 = (rois[0] + rois[3] + rois[6]) / 3
plt.imshow(p0)