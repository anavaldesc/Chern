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
import imutils
from collections import deque
from scipy.linalg import pinv, lu, solve
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
date = 20170726
sequence = 30
redo_prepare = True
fringe_removal = True
sequence_type = 'thermal_rabi_flop'
x0 = [233, 393]
x1 = [319, 174]
x0 = [243, 320]
x1 = [307, 541] # center location of atomic states involved in Rabi flop
wx = 120
wy = 120
theta = 9
folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)

Raman_pulse_time = []
fracs = []
probe_list = deque([], 20)
ods = []


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
                
                if fringe_removal:
                    probe_list.append(probe)
                    opt_ref  = fringeremoval([atoms], probe_list,
                                 mask='all', method='svd')
                    od = -np.log(((atoms < 1) + atoms) / 
                                ((opt_ref[0] < 1) + opt_ref[0]))
                
                else:
                    
                    od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
                
#                od = imutils.rotate_bound(od, theta)
                
  
                od0 = od[x0[0]-wx:x0[0]+wx, x0[1]-wy:x0[1]+wy].T
                od1 = od[x1[0]-wx:x1[0]+wx, x1[1]-wy:x1[1]+wy].T
                ods.append(od1)
#                fraction = od0 / (od0 + od1)
#                frac_row = []
                
#                for i in range(wx):
#                    frac_row.append(fraction[:,i:i+2].mean())
#                fracs.append(frac_row)
#                print(fracs)
    #        fracs = np.array(fracs)
                
#                plt.imshow(od.T, vmin=0, vmax=1)
#                plt.show()
                
        df = pd.DataFrame()
        df['Raman_pulse_time'] = Raman_pulse_time
        fracs = np.array(fracs)
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

except KeyError:
    print('Empty data file')        


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


