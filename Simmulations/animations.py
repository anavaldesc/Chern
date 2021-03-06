# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:29:12 2017

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
import TDSE


def H_RashbaRF(t, qx, qy, Omega1, Omega2, Omega3, delta1, delta2, delta3):
    
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


Omega1 = 7 / 2#6.04 /2 / 1.
Omega2 = 7 / 2# /2 / 1.
Omega3 = 7 / 2#6.5 /2 / 1.
delta1 = 4 * 1.839*0
delta2 = 0
delta3 = 4 * 1.839*0
kx = 0
ky = 0
Psi0 = [0, 0, 1]
t0 = 0
t1 = 600e-3#2/np.min([Omega1, Omega2, Omega3])
dt = (t1 - t0) / 3e2
Psi_vec = np.zeros((50, 50, 301, 3), dtype='complex')

kvec  = np.linspace(-2, 2, 50) * 1.7



for i, kx in enumerate(kvec):
    for j, ky in enumerate(kvec):

        args = [TDSE.H_RashbaRF, kx, ky, Omega1, Omega2, Omega3, delta1, delta2, delta3]
        (t_result, Psi_result) = TDSE.ODE_Solve(Psi0, t0, t1, dt, args)        
        Psi_vec[i ,j ,: , :] = np.abs(Psi_result) ** 2

for i in range(3):
    plt.plot(t_result*1e3, np.abs(Psi_result[:,i]) ** 2)
#SimplePlot(t_result, Psi_result)
plt.axis('Tight')
plt.xlabel('Pulse time [us]')
plt.ylabel('Fraction')


#%%
psi_stack = []
for i in range(3):
    psi_stack.append(Psi_vec[:,:,:,i])
#psi_stack.append(np.sum(np.abs(Psi_vec), axis=3))
psi_stack = np.array(psi_stack)
psi_stack = psi_stack.reshape((50*3, 50, 301))
plt.imshow(np.abs(psi_stack[:,:,100].T))

#%%
from Figtodat import fig2img
from images2gif import writeGif

# Compile into GIF
figure = plt.figure()
plot   = figure.add_subplot(111)
plot.hold(False)
images=[]
for i in range(30):
    plot.imshow(np.abs(psi_stack[:,:,i].T), vmin= 0.0, vmax=1)
    im = fig2img(figure)
    images.append(im)

#writeGif("start_in_y.gif",images, duration=0.1, dither=0)
#%%
from matplotlib.gridspec import GridSpec

state = ['z state', 'x state', 'y state']
images=[]
figure = plt.figure()
gs = GridSpec(1,3)
for j in range(20):

    for i in range(3):
        plt.subplot(gs[i])
        plt.imshow(np.abs(Psi_vec[:,:,j,i]))
        plt.xlabel('q_x')
        plt.ylabel('q_y')
        plt.xticks([])
        plt.yticks([])
    #    plt.xticks(np.linspace(0, 49, 3), ['-2', '0', '2'])
    #    plt.yticks(np.linspace(0, 49, 3), ['-2', '0', '2'])
        plt.title(state[i])
        plt.hold(False)
        
    im = fig2img(figure)
    images.append(im)

writeGif("pretty.gif",images, duration=0.1, dither=0)
#    plt.axis('Tight')

#    figure = plt.figure()
#    plot   = figure.add_subplot(111)
#    plot.hold(False)

#Psi_vec = np.array(Psi_vec)
#Psi_reshape = Psi_vec.reshape((50, 50, 301, 3))