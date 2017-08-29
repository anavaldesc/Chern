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
from collections import deque
from scipy.linalg import pinv, lu, solve

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
    
def fringeremoval(img_list, ref_list, mask='all', method='svd'):

    nimgs = len(img_list)
    nimgsR = len(ref_list)
    xdim = img_list[0].shape[0]
    ydim = img_list[0].shape[1]

    if mask == 'all':
        bgmask = np.ones([ydim, xdim])
        # around 2% OD reduction with no mask
    else:
        bgmask = mask

    k = (bgmask == 1).flatten(1)

    # needs to be >float32 since float16 doesn't work with linalg

    R = np.dstack(ref_list).reshape((xdim*ydim, nimgsR)).astype(np.float64)
    A = np.dstack(img_list).reshape((xdim*ydim, nimgs)).astype(np.float64)

    # Timings: for 50 ref images lasso is twice as slow
    # lasso 1.00
    # svd 0.54
    # lu 0.54

    optref_list = []

    for j in range(A.shape[1]):

        if method == 'svd':
            b = R[k, :].T.dot(A[k, j])
            Binv = pinv(R[k, :].T.dot(R[k, :]))  # svd through pinv
            c = Binv.dot(b)
            # can also try linalg.svd()

        elif method == 'lu':
            b = R[k, :].T.dot(A[k, j])
            p, L, U = lu(R[k, :].T.dot(R[k, :]))
            c = solve(U, solve(L, p.T.dot(b)))

        elif method == 'lasso':
            lasso = Lasso(alpha=0.01)
            lasso.fit(R, A)
            c = lasso.coef_

        else:
            raise Exception('Invalid method.')

        optref_list.append(np.reshape(R.dot(c), (xdim, ydim)))

    return optref_list



camera = 'XY_Flea3'
date = 20170728
sequence = 15
redo_prepare = True
sequence_type = 'thermal_rabi_flop'

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

Raman_pulse_time = []
probe_list = deque([], 40)
od_list = []
integrated_od = []


try:
    with h5py.File('results/' + outfile, 'r') as f:
        f['data']
except KeyError:
    redo_prepare = True
except IOError:
    redo_prepare = True

if redo_prepare:
#make probe list firts
    print('Preparing {} probe list...'.format(sequence_type))
    for r, file in matchfiles(folder):
        with h5py.File(os.path.join(r, file), 'r') as h5_file:
            try:
                img = h5_file['data']['images' + camera]['Raw'][:]
                probe = img[1] - img[2]
                probe_list.append(probe)
                
            except KeyError:
                
                print('bad shot')
            
        probe = img[1] - img[2]
        probe_list.append(probe)


    print('Preparing {} data...'.format(sequence_type))
    for r, file in matchfiles(folder):
        with h5py.File(os.path.join(r, file), 'r') as h5_file:
            img = h5_file['data']['images' + camera]['Raw'][:]
            
            attrs = h5_file['globals'].attrs
            Raman_pulse_time.append(attrs['Raman_pulse_time'])            
            
        img = np.float64(img)
        atoms = img[0] - img[2]
        probe = img[1] - img[2]
        
        probe_list.append(probe)
        opt_ref  = fringeremoval([atoms], probe_list,
                                 mask='all', method='svd')
        od = -np.log(((atoms < 1) + atoms) / ((opt_ref[0] < 1) + opt_ref[0])).T
        od_list.append(od)  
        integrated_od.append(np.nansum(od))

#        plt.imshow(od, vmin=0, vmax=1)
#        plt.show()
    df = pd.DataFrame()
#    df['od'] = od
#    df['p1'] = p1
#    df['p2'] = p2
    df['Raman_pulse_time'] = Raman_pulse_time
    df['integrated_od'] = integrated_od
#    df['delta_xyz'] = delta_xyz

    df = df.dropna()
    df = df.sort_values(by='Raman_pulse_time')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')

        

#%%
from Figtodat import fig2img
from images2gif import writeGif

figure = plt.figure()
plot   = figure.add_subplot(111)
plot.hold(False)
indices = df.index
images=[]

plt.imshow(od_list[indices[1]], vmin=0, vmax=1)
od = od_list[indices[1]].T
blob =  dbf.blob_detect(od,show=True,max_sigma=100, min_sigma = 90,
                        num_sigma=5, threshold=.2)
y_blob, x_blob = blob[0][0:2]
print(x_blob, y_blob)
#%%
sorted_roi = []
x_0 = [x_blob, y_blob]
for idx in indices[0:50]:                        

    od = od_list[idx].T
    od0 = od[x0[0]-wx:x0[0]+wx, x0[1]-wy:x0[1]+wy]
    sorted_roi.append(od0)

plt.imshow(sorted_roi[49])
plt.show()

#%%
n = 50
dt = 15.102041 - 5.0
dt = dt*1e-6
freqs = np.fft.fftfreq(n, dt)
psd_arr = []
for i in range(0, 50):
    psd_vec  = []
    for j in range(1):
        pops = (sorted_roi[:, 180, i])
        fpops = np.fft.fftshift(np.fft.fft(pops - pops.mean()))
        psd = np.abs(fpops)**2
        psd /= psd.max()
        psd_vec.append(psd[n/2::])
        
    
    plt.plot(freqs, psd)
    plt.xlim([0, freqs.max()])
    psd_arr.append(psd_vec)

psd_arr = np.array(psd_arr)
plt.pcolormesh(psd_arr[:,0,:], cmap='Greys')
plt.axis('Tight')
plt.ylim([0, 50])
plt.xticks([])
plt.yticks([])
plt.xlabel('q_x')
plt.ylabel('Frequency')
#%%
for i in df.index[1::2]:

    plot.imshow(od_list[i], vmin= 0.0, vmax=1)
    im = fig2img(figure)
    images.append(im)

writeGif("rabi_flop.gif",images, duration=0.3, dither=0)