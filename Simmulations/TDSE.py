# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:24:31 2016

@author: banano
"""

import scipy
import scipy.linalg
from scipy.integrate import ode
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
#import scipy.fftpack as sf
#import matplotlib.gridspec as gridspec
from numpy.linalg import eigvalsh
import cmath

def TDSE_Evolve(H, psi, t0, t1, dt, *args):
    """
    Evolves the TDSE from t0 (the initial condition) to t1 with timesteps of size dt, or
    a time step that splits the interval into equal sized bins
    
    H : the systems Hamiltonain, a function which takes as parameters H(t, **argvs)
    """
    
    steps = np.ceil((t1 - t0)/ dt)
    dt_correct = (t1 - t0)/steps # True timestep
    
    grid = np.linspace(t0, t1, num=steps)
    
    for t in grid:
        psi = np.einsum("ij,j",scipy.linalg.expm(-1.0j*dt_correct*H(t, *args)),psi)
    
    return psi

#
# Wrapper for builtin ode solver
# 

def RHS(t, Psi_tuple, args):
    """
    This function generically computes the right-hand-side for our 
    TDSE solver.
    
    t : time
    
    Psi_tuple : components of the wavefunction at time t
            
    args : additional arguments to be passed to H
    
    args[0] : function to generate hamiltonian matrix at time t
        H has calling convention H(t, *args)

    I did this this way rather than using H, *args in the calling header to overcome 
    a bug in the interface to the underlying fortran.
    """
    
    Hamiltonian = args[0]
    args=args[1:]
        
    Psi = np.array(Psi_tuple)
                
    return np.einsum('ij,j', Hamiltonian(t, *args)/(1.0J), Psi)

def ODE_Solve(Psi0, t0, t1, dt, args):
    """
    Solve the desired Shcrodinger based ODE 
        
    Psi0: initial wavefunction
    
    t0: initial time
    
    t1: final time
    
    dt: time step in returned paramters
    
    args: a tuple of arguments to be passed on to RHS
    
    """
        
    Psi_result = []
    t_output = []

    # Initialisation:

    solver = ode(RHS)

    backend = "zvode"
    solver.set_integrator(backend)  # nsteps=1
    
    solver.set_initial_value(Psi0, t0)
    solver.set_f_params(args)
        
    Psi_result.append(Psi0)
    t_output.append(t0)

#    while solver.successful() and solver.t < t1:
#        solver.integrate(solver.t + dt, step=1)
#
#        Psi_result.append(solver.y)
#        t_output.append(solver.t)
    
    t = t0
    
    while solver.successful() and t < t1:
   
        solver.integrate(t+dt)
#        print t
        Psi_result.append(solver.y)
        t_output.append(t+dt)
        t += dt
        
    return np.array(t_output), np.array(Psi_result)

def SimplePlot(t_result, Psi_result, ax = None):
    """
    Plot data of the form we are creating
    
    ax : if passed, this will ask for the traces to be added to an existing axis,
        otherwise a new one is created.
    """
    if ax is None:
        fig = plt.figure(0,figsize=(6,5));
        gs = mpl.gridspec.GridSpec(1, 1)
        gs.update(left=0.21, right=0.98, hspace=0.05, top=0.9)
        ax = fig.add_subplot(gs[0])

    Traces = Psi_result.shape
    
    for trace in range(Traces[1]):
        ax.plot(t_result*1e3, np.abs(Psi_result[:,trace])**2)

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Population')
#    ax.set_ylim([0,1])


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
#    k1_x = np.cos(2 * np.pi / (360./135))
#    k1_y = -np.sin(2 * np.pi / (360./135))
#    k2_x = np.cos(2 * np.pi  / 1.6)
#    k2_y = -np.sin(2 * np.pi / 1.6)
#    k3_x = np.cos(2 * np.pi)
#    k3_y = -np.sin(2 * np.pi)
    
    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2+delta1 + delta3, 1.0j*Omega1, Omega3], 
          [-1.0j*Omega1, (qx+k2_x)**2 + (qy+k2_y)**2-delta1+delta2, -1.0j*Omega2], 
          [Omega3, 1.0j*Omega2, (qx+k3_x)**2 + (qy+k3_y)**2-delta2-delta3]])
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
    
    
#%%
'''
this chunk is to plot 2D fringes for a given Ramsey wait time
'''
 
Omega = 3.0
Omega_big = 3*Omega
delta = 0
k = 0.1
t0 = 0.0 # Initial time
psi_final = []
psi_initial = []
phi0 = []
phi1 = []
phi2 = []
kvec = np.linspace(-1, 1, 60)
qx = 0.01
qy = 0
E = []
#for t in np.linspace(0.1, 1, 2):
t1 = 1.0 / ( Omega_big) / 8 * 1e2# free evolution time plus pi/2 pulse time

#t1 = t / Omega

for qx in kvec:
    for qy in kvec:

        dt = (t1 - t0) / 1e1# 
        t_wait = t1 - 1.0 / ( Omega_big) / 8
        Psi0 = np.array([1,0, 0])
        Psi0 = np.linalg.eigh(H_RashbaRF(t, qx, qy, Omega, Omega , Omega))[1]
        Psi0 = Psi0[:,0]
        E.append(np.linalg.eigh(H_RashbaRF(t, qx, qy, Omega, Omega, Omega))[0])
        args = [H_RashbaRamsey, qx, qy, Omega, Omega_big, t_wait]
        t_result, psi_result = ODE_Solve(Psi0, t0, t1, dt, args)
        #plt.plot(t_result, np.imag(psi_result[:,0]))
    #    for i in range(3):
    #        plt.plot(t_result, np.abs(psi_result[:, i])**2)
#        plt.plot(t_result, np.real(psi_result[:,0])**2)
        psi_final.append(psi_result[-1]) 
        psi_initial.append(Psi0)
        phi0.append(cmath.phase(Psi0[0]))
        phi1.append(cmath.phase(Psi0[1]))
        phi2.append(cmath.phase(Psi0[2]))

#%%
gs = gridspec.GridSpec(2, 3)
plt.figure(figsize=(11, 3*2))
plt.subplot(gs[0])
nk = len(kvec)
psi_final = np.array(psi_final)
psi_final = psi_final.reshape(nk, nk, 3)

psi_initial = np.array(psi_initial)
psi_initial = psi_initial.reshape(nk, nk, 3)
titles = ['z state fraction', 'x state fraction', 'y state fraction']

for i in range(3):
    plt.subplot(gs[0,2-i])
    plt.pcolormesh(np.abs(psi_final[:,:,i])**2, cmap='Greys')
    plt.title(titles[i] + ' final')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('qx')
    plt.ylabel('qy')
    
    plt.subplot(gs[1,2-i])
    plt.pcolormesh(np.abs(psi_initial[:,:,i])**2, cmap='Greys')
    plt.title(titles[i] + ' initial')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('qx')
    plt.ylabel('qy')
    
#%%

phi0 = np.array(phi0).reshape(nk, nk)
phi1 = np.array(phi1).reshape(nk, nk)
phi2 = np.array(phi2).reshape(nk, nk)

plt.imshow(phi0 + phi1 + phi2)
#%%    
'''
Rashba eigenstates vs k
'''


#%%
#
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

'''
This chunk looks at fringes with 1D soc
'''
Omega = 1
Omega_big = 5*Omega
delta = 0
k = 0.1
t0 = 0.0 # Initial time
psi_final = []
kvec = np.linspace(-1, 1, 10)
for k in kvec:
    
    for t in np.linspace(0.1, 10, 10):
    
        t1 = t / Omega
        #t1 = 0.5
        dt = (t1 - t0) / 5e3# T
        t_wait = t1 - 1.0 / ( Omega_big) / 8
        Psi0 = np.array([1,0])
        Psi0 = np.linalg.eigh(H_SOC(0, k, Omega, delta))[1][:,0]
        args = [H_Ramansey, k, Omega, Omega_big, delta, t_wait]
        t_result, psi_result = ODE_Solve(Psi0, t0, t1, dt, args)
        plt.plot(t_result, np.imag(psi_result[:,1]))
        plt.plot(t_result, np.abs(psi_result[:,1])**2)
        plt.plot(t_result, np.real(psi_result[:,1])**2)
        psi_final.append(np.abs(psi_result[-1])**2)

plt.plot(t_result, np.abs((psi_result[:,1])**2))
plt.xlim([t_wait, t1])

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