# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:24:31 2016

@author: banano
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import eigvalsh
from matplotlib import gridspec
from scipy.linalg import expm


def eigen(func, i, *args):
#    
    eigenval = eigvalsh(func(*args))[i]
    
    return eigenval
    

def H_Rabi(t, Omega, delta=0, epsilon=0):
    
    Omega = Omega * 2 * np.pi
    epsilon = epsilon * 2 * np.pi
    delta = delta * 2 * np.pi
    
    H = np.array([[  delta-epsilon,0.0,0.0],[0.0,- epsilon,0.0],
                  [0.0,0.0, - delta+epsilon]],dtype=complex)
    H += Omega * Fx

    return H
    
    
def H_SOC(t, k, Omega, delta):

    
    Omega = Omega * 2 * np.pi
    delta = delta * 2 * np.pi 

    H = np.array([[(k-1)**2 + delta, Omega], 
                   [Omega, (k+1)**2-delta]], dtype=complex)
    
    return H
    
def H_Ramansey(t, k, Omega, Omega_big, delta, t_wait):

#    
#    Omega = Omega * 2 * np.pi
#    delta = delta * 2 * np.pi 
#    Omega_big = Omega_big * 2 * np.pi
    
    if t < t_wait:
        Omega = 0
        H = H_SOC(t, k, Omega, delta)
#        print('or here')


    elif t >= t_wait:# and t < t_wait*(1 + 1.0 / ( Omega_big) / 8):#1/Omega_big/np.sqrt(2)/4:
        Omega = Omega_big
        H = H_SOC(t, k, Omega, delta)
#        print('here')
    
    else:
        Omega = 0
        H = H_SOC(t, k, Omega, delta)
    
    
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

def H_RashbaRF_td(t, qx, qy, omega_zx, omega_xy, omega_yz, 
                  Omega_zx, Omega_xy, Omega_yz):
    
#    Omega1 = Omega1 * 2 * np.pi
#    Omega2 = Omega2 * 2 * np.pi
#    Omega3 = Omega3 * 2 * np.pi
    
    k1_x = np.cos(2 * np.pi / 3) * np.sqrt(2)
    k1_y = -np.sin(2 * np.pi / 3) * np.sqrt(2)
    k2_x = np.cos(2 * np.pi * 2/ 3) * np.sqrt(2)
    k2_y = -np.sin(2 * np.pi * 2/ 3) * np.sqrt(2)
    k3_x = np.cos(2 * np.pi)
    k3_y = -np.sin(2 * np.pi)
    omega_RF = 0
#    k1_x = np.cos(2 * np.pi / (360./135))
#    k1_y = -np.sin(2 * np.pi / (360./135))
#    k2_x = np.cos(2 * np.pi  / 1.6)
#    k2_y = -np.sin(2 * np.pi / 1.6)
#    k3_x = np.cos(2 * np.pi)
#    k3_y = -np.sin(2 * np.pi)
    H = np.array([[-omega_zx, 0, 0], [0, 0, 0], [0, 0, omega_xy]], 
                 dtype='complex')
    H += [[0, Omega_zx * np.cos((omega_zx - omega_RF) * t) , 0], 
          [Omega_zx * np.cos((omega_zx - omega_RF) * t) , 0, 0],     
          [0, 0, 0]]
    H += [[0, 0, 0],
          [0, 0, Omega_xy * np.cos(omega_xy * t)],
          [0, Omega_xy * np.cos(omega_xy * t), 0]]  
    H += [[0, 0,  Omega_yz * np.cos(omega_yz * t)],
           [0, 0, 0],
           [Omega_yz * np.cos(omega_yz * t), 0, 0]]
    
    H += [[(qx+k1_x)**2 + (qy+k1_y)**2, 0, 0], 
          [0, (qx+k2_x)**2 + (qy+k2_y)**2, 0], 
          [0, 0, (qx+k3_x)**2 + (qy+k3_y)**20]]


    return H 
    

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
    
def H_RashbaRamsey(t, qx, qy, Omega, Omega_big, t_wait):
    
    if t < t_wait:
        Omega = 0
        H = H_RashbaRF(t, qx, qy, Omega, Omega , Omega)
#        print('or here')


    elif t >= t_wait:# and t < t_wait*(1 + 1.0 / ( Omega_big) / 8):#1/Omega_big/np.sqrt(2)/4:
        Omega = Omega_big
        H = H_RashbaRF(t, qx, qy, Omega, Omega, Omega)
#        print('here')
    
    else:
        Omega = 0
        H = H_RashbaRF(t, qx, qy, Omega, Omega, Omega)
    
    
    return H
    
def H_Rashba(t, qx, qy, Delta_z):
        
    sigma_x = np.array([[0, 1], [1, 0]], dtype='complex')
    sigma_y = np.array([[0, -1j], [-1j, 0]], dtype='complex')
    sigma_z = np.array([[1, 0], [0, -1]], dtype='complex')
    sigma_0 = np.array([[1, 0], [0, 1]], dtype='complex')
    
    H = (qx**2 + qy**2) * sigma_0
    H += sigma_x * qy - sigma_y * qx + Delta_z * sigma_z 
    H = np.array(H, dtype='complex')
    
    return H

def H_1Dsoc(t, qx=0, Omega=0, delta=0):
    
    H = [[(qx-1)**2 + delta, Omega], [Omega , (qx+2)**2]]
    H = np.array(H, dtype='complex')
    
    return H

def H_1Dsoc_ramp(t, qx=0, Omega=0, delta=0, ramp_rate=1):
    
    if t < 1 / ramp_rate:
    
        H = [[(qx-1)**2 + delta, Omega * t * ramp_rate], 
             [Omega * t * ramp_rate, (qx+2)**2 - delta]]
        H = np.array(H, dtype='complex')
        
    else:
        
        H = H_1Dsoc(t, qx, Omega, delta)
        
    
    return H



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


'''
Start with simple case, simulate 1D soc eigenstate preparation
'''

ramp_rate = 0.01
t = np.linspace(0, 1 / ramp_rate, 1e4) * 1.15
Omega = 3
qx = -1
psi0 = [0, 1]
kwargs = [qx, Omega, 0, 0.01]
psi_list, P_list, U_list = evolve(t, H_1Dsoc_ramp, psi0, kwargs)
psi_eigen = np.linalg.eigh(H_1Dsoc(t, qx, Omega, 0))[1]

plt.plot(t, P_list)
plt.plot(t[-1], np.abs(psi_eigen[:,0])[0]**2, 'o')
plt.plot(t[-1], np.abs(psi_eigen[:,0])[1]**2, 'o')

#%%

'''
Plot Rashba eigenstates
'''

n = int(1e2)
kvec = np.linspace(-1, 1, n)
Delta_z = 0
psi_rashba = []

for qx in kvec:
    for qy in kvec:

        Psi0 = np.linalg.eigh( H_Rashba(t, qx, qy, Delta_z))[1]
        Psi0 = Psi0[:,0]
        psi_rashba.append(Psi0)

gs = gridspec.GridSpec(1, 2)
plt.figure(figsize=(11, 3))
plt.subplot(gs[0])
nk = len(kvec)
psi_rashba = np.array(psi_rashba).reshape(nk, nk, 2)
#psi_final = psi_final.resha/pe(nk, nk, 3)


titles = ['spin up fraction', 'spin down fraction']

for i in range(2):
    plt.subplot(gs[i])
    plt.pcolormesh(np.real(psi_rashba[:,:,i]), cmap='Greys')
#    plt.title(titles[i] + ' final')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('qx')
    plt.ylabel('qy')

#some numbers seem to bave a global pi phase shift, no big deal...
#%%


'''
First lets see if time dependent Rashba Hamiltonian works...
'''
Omega = 2.
Omega1 = Omega
Omega2 = Omega
Omega3 = Omega
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
t = np.linspace(0, 0.5, 1e4)
#kwargs = [qx, qy, omega_zx, omega_xy, omega_yz, Omega_zx, Omega_xy, Omega_yz]
#kwargs_full = [qx, qy, 2* Omega1, 2* Omega, 2*Omega3, omega1, omega2, 
#             omega3, E1, E2, E3]
P_full = evolve(t, H_RashbaRF_full, psi0, args_full)[1]
P_Floquet = evolve(t, H_RashbaRF, psi0, args_floquet)[1]
plt.plot(t, P_full)
plt.plot(t, P_Floquet, '--')

# it does! Here I'm comparing FLoquet Hamiltonian with full time dependentent one
    
#%%
'''
Prepare one adiabatic Rashba eigenstate before moving onto crazy grids
'''

n = int(1e2)
kvec = np.linspace(-1, 1, n)
Delta_z = 0
psi_rashba_rf = []
Omega = 4

for qx in kvec:
    for qy in kvec:

        Psi0 = np.linalg.eigh( H_RashbaRF(t, qx, qy, Omega, Omega*0, Omega*0))[1]
        Psi0 = Psi0[:,0]
        psi_rashba_rf.append(Psi0)

gs = gridspec.GridSpec(1, 3)
plt.figure(figsize=(11, 3))
plt.subplot(gs[0])
nk = len(kvec)
psi_rashba_rf = np.array(psi_rashba_rf).reshape(nk, nk, 3)
#psi_final = psi_final.resha/pe(nk, nk, 3)


titles = ['z state', 'x state', 'y state']

for i in range(3):
    plt.subplot(gs[i])
    plt.pcolormesh(np.abs(psi_rashba_rf[:,:,i])**2, cmap='Greys')
    plt.title(titles[i])
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('qx')
    plt.ylabel('qy')
    



#'''
#This chunk looks at a given q state as a function of time
#'''
#
#Omega = 3.0
#Omega_big = 3*Omega
#delta = 0
#k = 0.1
#t0 = 0.0 # Initial time
#psi_final = []
#phi0 = []
#phi1 = []
#phi2 = []
#kvec = np.linspace(-1, 1, 100)
#qx = -1
#qy = -1
#E = []
##for t in np.linspace(0.1, 1, 2):
#tvec = np.linspace(0.1, 30, 100)
#
##t1 = t / Omega
#
#for t1 in tvec:
#
#    dt = (t1 - t0) / 5e4# T
#    t_wait = t1 - 1.0 / ( Omega_big) / 8
#    Psi0 = np.array([1,0, 0])
#    Psi0 = np.linalg.eigh(H_RashbaRF(0, qx, qy, Omega, Omega , Omega))[1]
#    Psi0 = Psi0[:,0]
#    E.append(np.linalg.eigh(H_RashbaRF(0, qx, qy, Omega, Omega, Omega))[0])
#    args = [H_RashbaRamsey, qx, qy, Omega, Omega_big, t_wait]
#    t_result, psi_result = ODE_Solve(Psi0, t0, t1, dt, args)
#    #plt.plot(t_result, np.imag(psi_result[:,0]))
##    for i in range(3):
##        plt.plot(t_result, np.abs(psi_result[:, i])**2)
##        plt.plot(t_result, np.real(psi_result[:,0])**2)
#    psi_final.append(psi_result[-1])  
#    phi0.append(cmath.phase(Psi0[0]))
#    phi1.append(cmath.phase(Psi0[1]))
#    phi2.append(cmath.phase(Psi0[2]))
#
#
#gs = gridspec.GridSpec(1, 3)
#plt.figure(figsize=(12, 3))
#plt.subplot(gs[0])
#nk = len(kvec)
#psi_final = np.array(psi_final)
##psi_final = psi_final.reshape(nk, nk, 3)
#titles = ['z state fraction', 'x state fraction', 'y state fration']
#
#psd_vec = []
#N = len(psi_final[:,0])
#d = tvec[1] - tvec[0]
##d = d*1e-6
#freqs = np.fft.fftfreq(N, d)[0:N/2]*1e3
#
#for i in range(3):
##    plt.subplot(gs[0,i])
##    plt.plot(tvec, np.abs(psi_final[:,i])**2)
##    plt.title(titles[i])
###    plt.colorbar()
##    plt.xticks([t0, t1])
##    plt.yticks([0, 0.5, 1])
##    plt.xlabel('t wait')
##    plt.ylabel('fraction')
#
#
#    plt.subplot(gs[0,i])
#    psi = np.abs(psi_final[:,i])**2
#    fpops = np.fft.fftshift( np.fft.fft(psi - psi.mean()))
#    N = len(fpops)
#    psd = np.abs(fpops[N/2::])**2
#    plt.plot(freqs, psd)
#    psd_vec.append(psd/psd.max())
#    plt.ylabel('PSD')
#    plt.xlabel('Frequency [Hz/2pi]')
#    plt.axis('Tight')
##    
##    plt.subplot(gs[2,i])
##    angle = np.angle(fpops)
##    plt.plot(tvec, angle/(2 * np.pi))
#    
#
#
##psd = np.array(psd)
##psd /= psd.max()
#
##%%
#
#%%
#
#'''
#This chunk looks at fringes with 1D soc
#'''
#Omega = 1
#Omega_big = 5*Omega
#delta = 0
#k = 0.1
#t0 = 0.0 # Initial time
#psi_final = []
#kvec = np.linspace(-1, 1, 10)
#for k in kvec:
#    
#    for t in np.linspace(0.1, 10, 10):
#    
#        t1 = t / Omega
#        #t1 = 0.5
#        dt = (t1 - t0) / 5e3# T
#        t_wait = t1 - 1.0 / ( Omega_big) / 8
#        Psi0 = np.array([1,0])
#        Psi0 = np.linalg.eigh(H_SOC(0, k, Omega, delta))[1][:,0]
#        args = [H_Ramansey, k, Omega, Omega_big, delta, t_wait]
#        t_result, psi_result = ODE_Solve(Psi0, t0, t1, dt, args)
#        plt.plot(t_result, np.imag(psi_result[:,1]))
#        plt.plot(t_result, np.abs(psi_result[:,1])**2)
#        plt.plot(t_result, np.real(psi_result[:,1])**2)
#        psi_final.append(np.abs(psi_result[-1])**2)
#
#plt.plot(t_result, np.abs((psi_result[:,1])**2))
#plt.xlim([t_wait, t1])

#%%
#
#'''
#This chunk looks at simple TDSE of Raman pulses
#'''
#
#Omega1 = 6.04e3 /2 / 1.839
#Omega2 = 6.5e3 /2 / 1.839
#Omega3 = 7e3 /2 / 1.839 
#Psi0 = [0, 0, 1]
#t0 = 0
#t1 = 800e-6#2/np.min([Omega1, Omega2, Omega3])
#dt = (t1 - t0) / 3e2
#(t_result, Psi_result) = ODE_Solve(Psi0, t0, t1, dt, 
#                    (H_RashbaRF, 0, 0, Omega1, Omega2, Omega3))
#
#for i in range(3):
#    plt.plot(t_result*1e6, np.abs(Psi_result[:,i])**2)
##SimplePlot(t_result, Psi_result)
#plt.axis('Tight')
#plt.xlabel('Pulse time [us]')
#plt.ylabel('Fraction')
##%%
#
#kvec = np.linspace(-1, 1, 30)
#sp = []
#Omega = 1 / (2*np.pi)
#
#for k in kvec:
#    
#    (t_result, Psi_result) = ODE_Solve(Psi0, t0, t1, dt, 
#                        (H_SOC, k, Omega, delta))
#    pops = np.abs(Psi_result)**2
#    pops -= np.mean(pops, axis=0)
#    N = len(pops)
#    dt = t_result[2]-t_result[1]
#    xf = np.linspace(0.0, 1/(2 * dt), N  / 2 )
#    fpops = np.array([np.abs(sf.fft(pops[:, i])**2) for i in range(2)])    
#    fpops = fpops/np.max(fpops)
#    fpops = fpops[:, :int(N/2)].T
#    plt.plot(fpops)
##        for i in range(0,3):
##            fpops[:,i] = fpops[:,i]/np.max(fpops[:,i])
##        fpops = np.mean(fpops, axis=1)
#    sp.append(fpops)
#    
#sp = np.array(sp)
#
##%%
#plt.imshow(sp[:,:,0].T)
#plt.ylim([0, 10])
##%%
#kvec = np.linspace(-1, 1, 100)
#Omega = 1
#args = [0, kvec, Omega / (2 * np.pi), delta]
#
#plt.plot(kvec, eigenarray(H_SOC, 1, *args) - eigenarray(H_SOC, 0, *args))
#
##%%
#kvec = np.linspace(-1, 1, 30)
#sp = []
#Omega = 1 / (2*np.pi)
#
#for k in kvec:
#    
#    Omega1 = 5.0
#    Omega2 = 5.0
#    Omega3 = 5.0
#    Psi0 = [0, 1, 0]
#    t0 = 0.0
#    t1 = 5/np.min([Omega1, Omega2, Omega3])
#    dt = (t1 - t0) / 5e2
#    (t_result, Psi_result) = ODE_Solve(Psi0, t0, t1, dt, 
#                    (H_Rashba, 0, 0, Omega1, Omega2, Omega3))
#    pops = np.abs(Psi_result)**2
#    pops -= np.mean(pops, axis=0)
#    N = len(pops)
#    dt = t_result[2]-t_result[1]
#    xf = np.linspace(0.0, 1/(2 * dt), N  / 2 )
#    fpops = np.array([np.abs(sf.fft(pops[:, i])**2) for i in range(3)])    
#    fpops = fpops/np.max(fpops)
#    fpops = fpops[:, :int(N/2)].T
#    plt.plot(fpops)
##        for i in range(0,3):
##            fpops[:,i] = fpops[:,i]/np.max(fpops[:,i])
##        fpops = np.mean(fpops, axis=1)
#    sp.append(fpops)
#    
#sp = np.array(sp)
#
##%%
#kvec = np.linspace(-2, 2, 200)
#args = [0, 0, kvec, Omega, Omega, Omega]
#for i in range(3):
#    plt.plot(kvec, eigenarray(H_Rashba, i, *args))