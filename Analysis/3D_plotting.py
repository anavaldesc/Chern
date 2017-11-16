#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:00:57 2017

@author: banano
"""

import numpy as np
from scipy.linalg import eigh
from mayavi import mlab


def H_Rashba(qx, qy, Delta_z):
    
    sigma_0 = np.array([[1, 0], [0, 1]], dtype='complex')
    sigma_x = np.array([[0, 1], [1, 0]], dtype='complex')
    sigma_y = np.array([[0, -1j], [-1j, 0]], dtype='complex')
    sigma_z = np.array([[1, 0], [0, -1]], dtype='complex')

    H = (qx**2 + qy**2) * sigma_0
    H += sigma_x * qy - sigma_y * qx + Delta_z * sigma_z 
    
    return H

q_vec = np.linspace(-0.5, 0.5, 2**6)

Delta_z = 0
energies = []
for qx in q_vec:
    for qy in q_vec:
        energies.append(eigh(H_Rashba(qx, qy, Delta_z))[0])
        
energies = np.array(energies)
energies = energies.reshape(len(q_vec), len(q_vec), 2)



[qx, qy] = np.meshgrid(q_vec, q_vec)
s = mlab.mesh(qx, qy, energies[:,:,0])
s2 = mlab.mesh(qx, qy, energies[:,:,1])

mlab.show()

#x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
#s = np.sin(x*y*z)/(x*y*z)
#mlab.pipeline.volume(mlab.pipeline.scalar_field(s), vmin=0, vmax=0.9)
#mlab.show()