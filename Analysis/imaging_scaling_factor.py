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
date = 20170710
sequence = 88
redo_prepare = False
sequence_type = 'z_0kr_position'

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

#Raman_pulse_time = []
x_val = []
y_val = []
od_m1 = []


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
#            print('banana')
            img = h5_file['data']['images' + camera]['Raw'][:]
            
            attrs = h5_file['globals'].attrs
            
        atoms = img[0] - img[2]
        probe = img[1] - img[2]
        img = np.float64(img)
        od = -np.log(((img[0] < 1) + img[0]) / ((img[1] < 1) + img[1])).T
        od_m1.append(od)
        blob =  dbf.blob_detect(od,show=True,max_sigma=70, min_sigma = 50,
                         num_sigma=5, threshold=.2)
        y_blob, x_blob = blob[0][0:2]
        x_val.append(x_blob)
        y_val.append(y_blob)
#        plt.imshow(od, vmin=0, vmax=4)
#        plt.show()
    df = pd.DataFrame()
    df['x_val'] = x_val
    df['y_val'] = y_val
    df['od'] = np.array(od_m1).tolist()

#    df['p1'] = p1
#    df['p2'] = p2
#    df['Raman_pulse_time'] = Raman_pulse_time
#    df['delta_xyz'] = delta_xyz

    df = df.dropna()
#    df = df.sort_values(by='Raman_pulse_time')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')

xmean_m1 = df['x_val'].mean()
ymean_m1 = df['y_val'].mean()

od =[]
for i in range(len(df)):
    od.append(np.array(df['od'][i]))
odmean_m1 = np.mean(od, axis=0)
    

#odmean_m1 = df['od'].mean()

print('xpos, ypos = ' + '[{},{}]'.format(xmean_m1, ymean_m1) )#', ' +  '{}'.format(ymean)))


#%%
sequence = 87
redo_prepare = False
sequence_type = 'x_0kr_position'

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

#Raman_pulse_time = []
x_val = []
y_val = []
od_0 = []


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
#            print('banana')
            img = h5_file['data']['images' + camera]['Raw'][:]
            
            attrs = h5_file['globals'].attrs
            
        atoms = img[0] - img[2]
        probe = img[1] - img[2]
        img = np.float64(img)
        od = -np.log(((img[0] < 1) + img[0]) / ((img[1] < 1) + img[1])).T
        od_0.append(od)
        blob =  dbf.blob_detect(od,show=True,max_sigma=70, min_sigma = 50,
                         num_sigma=5, threshold=.2)
        y_blob, x_blob = blob[0][0:2]
        x_val.append(x_blob)
        y_val.append(y_blob)
#        plt.imshow(od, vmin=0, vmax=4)
#        plt.show()
    df = pd.DataFrame()
    df['x_val'] = x_val
    df['y_val'] = y_val
    df['od'] = np.array(od_0).tolist()

#    df['p1'] = p1
#    df['p2'] = p2
#    df['Raman_pulse_time'] = Raman_pulse_time
#    df['delta_xyz'] = delta_xyz

    df = df.dropna()
#    df = df.sort_values(by='Raman_pulse_time')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')

xmean_0 = df['x_val'].mean()
ymean_0 = df['y_val'].mean()
#odmean_0 = df['od'].mean()
od =[]
for i in range(len(df)):
    od.append(np.array(df['od'][i]))
odmean_0 = np.mean(od, axis=0)

print('xpos, ypos = ' + '[{},{}]'.format(xmean_0, ymean_0) )#

#%%
w = 160
n = 4

croppedod_0 = odmean_0.T[int(xmean_0)-w:int(xmean_0)+w, 
                         int(ymean_0)-w:int(ymean_0)+w].T

croppedod_m1 = odmean_m1.T[int(xmean_m1)-w:int(xmean_m1)+w, 
                         int(ymean_m1)-w:int(ymean_m1)+w].T
                         
croppedod_0 /= croppedod_0.max()
croppedod_m1 /= croppedod_m1.max()

dif = croppedod_0 / (croppedod_m1*1 + 0*croppedod_0)
dif = croppedod_m1 / (croppedod_m1*0 + 1*croppedod_0)

                         
plt.imshow(dif, vmin=np.nanmean(dif) - np.nanmean(dif),
           vmax=np.nanmean(dif) + np.nanmean(dif), interpolation='none')
plt.colorbar()

#%%
plt.imshow(croppedod_m1, interpolation='none')
plt.colorbar()

#croppedod_0 = odmean_0.T[10:40, 10:40]

#od_m1 = np.arra