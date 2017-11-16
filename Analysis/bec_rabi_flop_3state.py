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
from scipy.sparse.linalg import expm



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
sequence = 183# 151
redo_prepare = False
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
p0 = []
p1 = []
p2 = []
psum = []

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
            
            try:

                img = h5_file['data']['images' + camera]['Raw'][:]
                attrs = h5_file['globals'].attrs
                
                            
                img = np.float64(img)
                atoms = img[0] - img[2]
                probe = img[1] - img[2]
                od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe))
                iod.append(np.ma.masked_invalid(od).sum())
                
#                attrs = h5_file['results/rois_od'].1attrs
                Raman_pulse_time.append(attrs['Raman_pulse_time'])  
#                print(attrs[:])
                
            except Exception as e:
                
                print(e)
#                print('There are no {} images in this file'.format(camera))
            
            

            try:
                attrs = h5_file['results/rois_od'].attrs
                p0.append(attrs['roi_0'])
                p1.append(attrs['roi_1'])
                p2.append(attrs['roi_2'])
                psum.append(attrs['roi_0'] + attrs['roi_1'] + attrs['roi_2'])
            except:
                
                print('There are no rois in this shot')
#                
                
        df = pd.DataFrame()
        df['Raman_pulse_time'] = Raman_pulse_time
        df['integratedOD'] = iod
        df['p0'] = p0
        df['p1'] = p1
        df['p2'] = p2 
        df['psum'] = psum           
#            df['p{}'.format(i)] = p0
        fracs = np.array(fracs)
        ods = np.array(ods)
#        for i in range(3):
#            df['p{}'.format(2-i)] = ods[:,i]
    
    #            for i in range(50):
#        df['frac'] = fraction[:,1:1+2].mean()
    #            print(fraction[:,1:1+4].mean())
    #                print(fr action[:,i:i+4].mean())
    #    df['delta_xyz'] = delta_xyz
    
        df = df.dropna()
        df = df.sort_values(by='Raman_pulse_time')
#        df = df[ df.integratedOD > df.integratedOD.mean() * 1 ]
    #    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')

#except Exception as e:
#    print(e)
#    print('Empty data file')        


#%%
color = ['b', 'k', 'r']
for i in range(3):
    
    plt.plot(df['Raman_pulse_time'], df['p{}'.format(i)]/ df['psum'], color[i] + 'o')
    plt.xlabel('Raman pulse time')
    plt.ylabel('Fraction')
#plt.xlim([0, 500])
#%%

def evolve(t, H, psi0, kwargs):
    dt = np.diff(t)
    Hlist = np.array([H(ti, *kwargs) for ti in t[:-1]+dt/2])
    Ulist = []
    psiList = [psi0]
    Plist = [np.abs(psi0)**2]
    for dti, Hi in zip(dt, Hlist):
        Ui = expm(-1j*Hi*dti)
        psi = np.dot(Ui, psiList[-1])
        Ulist.append(Ui)
        psiList.append(psi)
        Plist.append(np.abs(psi)**2)
    Ulist = np.array(Ulist)
    psiList = np.array(psiList)
    Plist = np.array(Plist)
    return psiList, Plist, Ulist

def H_RashbaRF_full(t, ramp_rate, qx, qy, Omega1, Omega2, Omega3, 
                    omega1, omega2, omega3, E1, E2, E3, ramp):
    
    if ramp and t <= 1 / ramp_rate:
        
        Omega1 = Omega1 * 2 * np.pi * t * ramp_rate
        Omega2 = Omega2 * 2 * np.pi * t * ramp_rate
        Omega3 = Omega3 * 2 * np.pi * t * ramp_rate
        
    else:
        Omega1 = Omega1 * 2 * np.pi
        Omega2 = Omega2 * 2 * np.pi 
        Omega3 = Omega3 * 2 * np.pi 
        
    k1_x = np.cos(2 * np.pi / 3) * np.sqrt(2)
    k1_y = -np.sin(2 * np.pi / 3) * np.sqrt(2)
    k2_x = np.cos(2 * np.pi * 2/ 3) * np.sqrt(2)
    k2_y = -np.sin(2 * np.pi * 2/ 3) * np.sqrt(2)
    k3_x = np.cos(2 * np.pi)
    k3_y = -np.sin(2 * np.pi)
    
    k1_x = np.cos(2 * np.pi / (360./135))
    k1_y = -np.sin(2 * np.pi / (360./135))
    k2_x = np.cos(2 * np.pi  / 1.6)
    k2_y = -np.sin(2 * np.pi / 1.6)
    k3_x = np.cos(2 * np.pi)
    k3_y = -np.sin(2 * np.pi)
    
    H = np.array([[E1 + (qx+k1_x)**2 + (qy+k1_y)**2, 1.0j*Omega1*np.cos(omega1*t), Omega3*np.cos(omega3*t)], 
                  [-1.0j*Omega1*np.cos(omega1*t),E2 + (qx+k2_x)**2 + (qy+k2_y)**2, -1.0j*Omega2*np.cos(omega2*t)], 
                  [Omega3*np.cos(omega3*t), 1.0j*Omega2*np.cos(omega2*t),E3 + (qx+k3_x)**2 + (qy+k3_y)**2]])
    H = np.array(H, dtype='complex')

    return H 

def H_RashbaRF(t, qx, qy, Omega1, Omega2, Omega3, delta1=0, delta2=0, delta3=0):
    
    Omega1 = Omega1 * 2 * np.pi
    Omega2 = Omega2 * 2 * np.pi
    Omega3 = Omega3 * 2 * np.pi
    
    k1_x = np.cos(2 * np.pi / 3) * np.sqrt(2)
    k1_y = -np.sin(2 * np.pi / 3) * np.sqrt(2)
    k2_x = np.cos(2 * np.pi * 2/ 3) * np.sqrt(2)
    k2_y = -np.sin(2 * np.pi * 2/ 3) * np.sqrt(2)
    k3_x = np.cos(2 * np.pi)
    k3_y = -np.sin(2 * np.pi)
#    
    k1_x = np.cos(2 * np.pi / (360./135))
    k1_y = -np.sin(2 * np.pi / (360./135))
    k2_x = np.cos(2 * np.pi  / 1.6)
    k2_y = -np.sin(2 * np.pi / 1.6)
    k3_x = np.cos(2 * np.pi)
    k3_y = -np.sin(2 * np.pi)
    
#    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2+delta1 + delta3, 1.0j*Omega1, Omega3], 
#          [-1.0j*Omega1, (qx+k2_x)**2 + (qy+k2_y)**2-delta1+delta2, -1.0j*Omega2], 
#          [Omega3, 1.0j*Omega2, (qx+k3_x)**2 + (qy+k3_y)**2-delta2-delta3]])
    
    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2+delta1 + delta3, Omega1, Omega3], 
      [Omega1, (qx+k2_x)**2 + (qy+k2_y)**2-delta1+delta2, Omega2], 
      [Omega3, Omega2, (qx+k3_x)**2 + (qy+k3_y)**2-delta2-delta3]])
    H = np.array(H, dtype='complex')

    return H 

Omega = 3.2
Omega1 = Omega * 1.
Omega2 = Omega * 0.9
Omega3 = Omega * 1.1
E1 = 0
E2 = 224
E3 = 224 + 140
delta1 = 0
delta2 = 0
delta3 = 0
omega1 = E2 - E1
omega2 = E3 - E2
omega3 = E3 - E1
qx = 0
qy = 0
ramp_rate = 1
args_floquet = [qx, qy, Omega1, Omega2, Omega3, delta1, delta2, delta3]
args_full = [ramp_rate, qx, qy, 2* Omega1, 2* Omega2, 2*Omega3, omega1, omega2, 
             omega3, E1, E2, E3, False]

psi0 = np.array([1, 0, 0], dtype='complex')
t = df['Raman_pulse_time']*1e-3
#kwargs = [qx, qy, omega_zx, omega_xy, omega_yz, Omega_zx, Omega_xy, Omega_yz]
#kwargs_full = [qx, qy, 2* Omega1, 2* Omega, 2*Omega3, omega1, omega2, 
#             omega3, E1, E2, E3]
P_full = evolve(t, H_RashbaRF_full, psi0, args_full)[1]
P_Floquet = evolve(t, H_RashbaRF, psi0, args_floquet)[1]

#%%

label = ['z', 'x', 'y']
for i in range(3):
#    plt.plot(t*1e3, P_full[:,i], color[i])
    plt.plot(t*1e3, P_Floquet[:,i], color[i] + '--', label=label[i] + ' rwa')
plt.xlabel('Pulse time [hbar/E_R]')
plt.ylabel('Probability')
#plt.legend()

#sorted_fractions = []
#for i in range(wx):
#    sorted_fractions.append(df['frac{}'.format(i)])
#    plt.plot(df['Raman_pulse_time'], df['frac{}'.format(i)])
#    plt.ylim([0,1])
#    plt.xlabel('Raman pulse time [us]')
#    plt.ylabel('Fraction')
#plt.show()
#
#sorted_fractions = np.array(sorted_fractions)
#psd_vec = []
#for i in range(len(sorted_fractions[:,0])):
#    frac_ft = np.fft.fftshift( np.fft.fft(sorted_fractions[i]-sorted_fractions[i].mean()))
#    N = len(frac_ft)
#    psd = np.abs(frac_ft[N/2::])**2
##    plt.plot(psd)
#    psd_vec.append(psd/psd.max())
#
#psd = np.array(psd_vec)
#psd /= psd.max()
#N = len(sorted_fractions[:,0])
#fet = df.as_matrix().T[0]
#d = fet[1] - fet[0]
#d = d*1e-6
#freqs = np.fft.fftfreq(N, d)[0:N/2]*1e-3
#plt.pcolormesh(psd.T, vmin=0, vmax=1, cmap='YlGn')
#plt.xlabel('pixel')
#plt.ylabel('Frequency [kHz]')
#plt.yticks(range(0, N/2, N/10), ['%.1f'%freqs[::int(N/10)][i] for i in range(0,5)])
#plt.axis('Tight')


