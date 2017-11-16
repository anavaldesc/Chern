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



camera = 'ProEM'
camera = 'XY_Flea3'
date = 20170906
sequence = 63
redo_prepare = True
sequence_type = 'x_0kr_position'
crop = False
x_crop = 336#243 #[79, 328, 535], [80, 332, 178], [124, 375, 579]
y_crop = 190#323 # [279, 344, 252], [178, 244, 151], [231, 292, 202]
wx_crop = 60
wy_crop = 40
i = 0

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

#Raman_pulse_time = []
x_val = []
y_val = []


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
            od = -np.log(((a < 1) + a) / ((p < 1) + p))
        
        elif camera == 'XY_Flea3':
            isat = 539.393939394
            a = img[0] - img[2]
            p = img[1] - img[2]
        
            od = -np.log(((a < 1) + a) / ((p < 1) + p)) + (p - a) / isat
            od = od.T
#            plt.imshow(od)
        if crop:
            
            od = od[y_crop - wy_crop: y_crop + wy_crop, x_crop - wx_crop: x_crop + wx_crop]
            
            try:
            
                blob =  utils.blob_detect(od,show=False,max_sigma=20, min_sigma = 10,
                             num_sigma=5, threshold=.2)
                x_blob, y_blob = blob[0][0:2] 
                x_blob += x_crop - wx_crop
                y_blob += y_crop - wy_crop
                print[blob[0][0:2]]
            
          
            except Exception as e:
                
                y_blob, x_blob = [np.nan, np.nan]
#                print (y_blob, x_blob)
#                print (e)
    
        else:
            
            try:
            
                blob =  utils.blob_detect(od,show=True,max_sigma=30, min_sigma = 25,
                             num_sigma=5, threshold=.2)
                x_blob, y_blob = blob[0][0:2] 
                print(blob)
            
#                print(blob)
            except TypeError:
                
                 y_blob, x_blob = [np.nan, np.nan]
                
            
    
        x_val.append(x_blob)
        y_val.append(y_blob)
#        plt.imshow(od, vmin=0, vmax=4)
#        plt.show()
    df = pd.DataFrame()
    df['x_val'] = x_val
    df['y_val'] = y_val

#    df['p1'] = p1
#    df['p2'] = p2
#    df['Raman_pulse_time'] = Raman_pulse_time
#    df['delta_xyz'] = delta_xyz

    df = df.dropna()
#    df = df.sort_values(by='Raman_pulse_time')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')
    
xmean = np.nanmean(df['x_val'])
ymean = np.nanmean(df['y_val'])

print('xpos, ypos = ' + '[{},{}]'.format(xmean, ymean) )#', ' +  '{}'.format(ymean)))

#%%

#od = -np.log(((a < 1) + a) / ((p < 1) + p)) + (p - a) / isat
#plt.imshow(od[300:400, 200:300])
#cropped_od = od[368-50:368+50, 274-50:274+50]
#blob1 =  dbf.blob_detect(od,show=True,max_sigma=30, min_sigma = 25,
#                             num_sigma=5, threshold=.2)
#blob2 =  dbf.blob_detect(cropped_od,show=True,max_sigma=30, min_sigma = 25,
#                             num_sigma=5, threshold=.2)
##plt.show()
#plt.pcolormesh(od)