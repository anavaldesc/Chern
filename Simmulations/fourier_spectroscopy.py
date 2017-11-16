# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:29:12 2017

@author: banano
"""

import sys
sys.path.append('/Users/banano/Documents/UMD/Research/Rashba/Chern/Utils')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

#import TDSE

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

def H_RashbaRF(t, qx, qy, Omega1, Omega2, Omega3, delta1, delta2, delta3):
    
    Omega1 = Omega1 * 2 * np.pi
    Omega2 = Omega2 * 2 * np.pi
    Omega3 = Omega3 * 2 * np.pi
    
#    k1_x = np.cos(2 * np.pi / 3) * np.sqrt(2)
#    k1_y = -np.sin(2 * np.pi / 3) * np.sqrt(2)
#    k2_x = np.cos(2 * np.pi * 2/ 3) * np.sqrt(2)
#    k2_y = -np.sin(2 * np.pi * 2/ 3) * np.sqrt(2)
#    k3_x = np.cos(2 * np.pi)
#    k3_y = -np.sin(2 * np.pi)
#    
    k1_x = np.cos(2 * np.pi / (360./135))
    k1_y = -np.sin(2 * np.pi / (360./135))
    k2_x = np.cos(2 * np.pi  / 1.6)
    k2_y = -np.sin(2 * np.pi / 1.6)
    k3_x = np.cos(2 * np.pi)
    k3_y = -np.sin(2 * np.pi)
    
    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2+delta1 + delta3, 1.0j*Omega1, Omega3], 
          [-1.0j*Omega1, (qx+k2_x)**2 + (qy+k2_y)**2-delta1+delta2, -1.0j*Omega2], 
          [Omega3, 1.0j*Omega2, (qx+k3_x)**2 + (qy+k3_y)**2-delta2-delta3]])
    H = np.array(H, dtype='complex')

    return H 
    

    
def H_RashbaReal(t, qx, qy, Omega1, Omega2, Omega3, omega1, omega2, omega3, E1, E2, E3):
    
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
#%%
'''
Compare full time dependent Hamiltonian with RWA
'''

Omega1 = 4.
Omega2 = 4.
Omega3 = 4.
E1 = -23e3
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

args_rwa = [H_RashbaRF, qx, qy, Omega1, Omega2, Omega3, delta1, delta2, delta3]
args_real = [H_RashbaReal, qx, qy, 2* Omega1, 2* Omega2, 2*Omega3, omega1, omega2, 
             omega3, E1, E2, E3]

Psi0 = (1, 0, 0)
t0 = 0
t1 = 600e-3#2/np.min([Omega1, Omega2, Omega3])
dt = (t1 - t0) / 1e4
(t_rwa, Psi_rwa) = ODE_Solve(Psi0, t0, t1, dt, args_rwa)
(t_real, Psi_real) = ODE_Solve(Psi0, t0, t1, dt, args_real)
#%%
lines_rwa = ['b--', 'k--', 'r--']
lines_floquet = ['b-', 'k-', 'r-']
for i in range(3):
    plt.plot(t_rwa, np.abs(Psi_rwa[:,i])**2, lines_rwa[i], 
             t_real, np.abs(Psi_real[:,i])**2,lines_floquet[i])
#    plt.hold()
#    plt.plot(t_real, np.abs(Psi_real[:,i])**2,'-')
plt.ylim([0,1])
#%%


Omega1 = 7 / 1 * 1#6.04 /2 / 1.
Omega2 = 7 / 1 * 1# /2 / 1.
Omega3 = 7 / 1 * 1 #6.5 /2 / 1.
delta1 = 4 * 1.839*0
delta2 = 0
delta3 = 4 * 1.839*0
kx = 0
ky = 0
Psi0 = [0, 1, 0]
t0 = 0
t1 = 900e-3#2/np.min([Omega1, Omega2, Omega3])
dt = (t1 - t0) / 1e2
Psi_vec = np.zeros((50, 50, 101, 3), dtype='complex')

kvec  = np.linspace(-2, 2, 50) * 5

def k_evolve(Psi0, k_dim, t_final, n_steps, args):
    
#Psi0 = [0, 1, 0]
    t0 = 0
    t1 = t_final#2/np.min([Omega1, Omega2, Omega3])
    dt = (t1 - t0) / n_steps
    Psi_array = []
    kvec  = np.linspace(-2, 2, k_dim) * 5

    for i, kx in enumerate(kvec):
        Psi_row = []
        
        for j, ky in tqdm(enumerate(kvec)):
    
            args[1] = kx
            args[2] = ky
            (t_result, Psi_result) = ODE_Solve(Psi0, t0, t1, dt, args) 
#            evolve(t, H, psi0, kwargs)
            Psi_row.append(np.abs(Psi_result) ** 2)
        
        Psi_array.append(Psi_row)
        
    Psi_array = np.array(Psi_array)
    
    return t_result, Psi_array
            

Psi0 = (1, 0, 0)
k_dim = 30
t_final = 1200e-3
n_steps = 120*1e2

#t_result, Psi_array_rwa = k_evolve(Psi0, k_dim, t_final, n_steps, args_rwa)
#%%
t_result, Psi_array_real = k_evolve(Psi0, k_dim, t_final, n_steps, args_real)
#%%

#Psi_array = Psi_array_rwa
Psi_array = Psi_array_real


state = ['z state', 'x state', 'y state']
images=[]
figure = plt.figure()
gs = GridSpec(1,3)


for i in range(3):
    plt.subplot(gs[i])
    plt.imshow(np.abs(Psi_array[:,:,120,i]))
    plt.xlabel('q_x')
    plt.ylabel('q_y')
    plt.xticks([])
    plt.yticks([])
#    plt.xticks(np.linspace(0, 49, 3), ['-2', '0', '2'])
#    plt.yticks(np.linspace(0, 49, 3), ['-2', '0', '2'])
    plt.title(state[i])
#    plt.hold(False)
    
#%%
    
#plt.plot(Psi_vec[25, 25, :,0]) 
Psi_array = Psi_array_rwa

n = len(t_result)
dt = t_result[1] - t_result[0]
freqs = np.fft.fftfreq(n, dt)
psd_array = np.zeros((k_dim, k_dim, int(n / 2), 3))
#psd_arr = []
for i in range(0, k_dim):
    for j in range(k_dim):
        for k in range(3):
            
    
            pops = np.abs(Psi_array[i, j, :, k])
    #        pops = np.abs(Psi_array[i, 14, :, j])
            fpops = np.fft.fftshift(np.fft.fft(pops - pops.mean()))
            
            
            psd = np.abs(fpops[int(n / 2 )::])**2
            psd /= psd.max()
            psd_array[i, j, :, k] = psd
    
#    plt.plot(freqs, psd)
#    plt.xlim([0, freqs.max()])
#    psd_arr.append(psd_vec)
    
    
#%%    
s = psd_array.sum(axis=3)[:, :, 0:30]
from mayavi import mlab    
#s = psd_array.sum(axis=3)[0:20]
mlab.pipeline.volume(mlab.pipeline.scalar_field(s), vmin=0, vmax=1)
#mlab.show()

	
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='x_axes',
                            slice_index=10,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='y_axes',
                            slice_index=10,
                        )
#mlab.outline()
mlab.show()
#psd_arr = np.array(psd_array)
#state = ['z state', 'x state', 'y state']
#images=[]
#figure = plt.figure()
#gs = GridSpec(1,3)
#
#
#for i in range(3):
#    plt.subplot(gs[i])
#    plt.pcolormesh(psd_array[:,i,:].T)
##    plt.xlabel('q_x')
##    plt.ylabel('q_y')
##    plt.xticks([])
##    plt.yticks([])
##    plt.xticks(np.linspace(0, 49, 3), ['-2', '0', '2'])
##    plt.yticks(np.linspace(0, 49, 3), ['-2', '0', '2'])
#    plt.title(state[i])
##    plt.hold(False)
#    plt.ylim([0, 80])
#    #plt.ylim([0, 50])
#    plt.xticks([])
#    plt.yticks([])
#    plt.xlabel('q_x')
#
#
#
#plt.axis('Tight')
#plt.ylim([0, 80])
#plt.xticks([])
#plt.yticks([])
##plt.xlabel('q_x')
##plt.ylabel('Frequency')
##im = fig2img(figure)
#images.append(im)
#%%
