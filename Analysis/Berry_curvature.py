# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:11:52 2017

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from numpy.linalg import eigh

sigma_x = np.array([[0, 1], [1, 0]], dtype='complex')
sigma_y = np.array([[0, -1j], [-1j, 0]], dtype='complex')
sigma_z = np.array([[1, 0], [0, -1]], dtype='complex')
sigma_0 = np.array([[1, 0], [0, 1]], dtype='complex')
F_123 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype='complex')

def H_Rashba(qx, qy, Delta_z):
    
    H = (qx**2 + qy**2) * sigma_0
    H += sigma_x * qy - sigma_y * qx + Delta_z * sigma_z 
    H = np.array(H, dtype='complex')
    
    return H
    
def eigen(func, i, *args):
#    
    eigenval = eigvalsh(func(*args))[i]
    
    return eigenval

def eigenvec(func, i, *args):
    
    eigenvec = eigh(func(*args))[1][:,i]
    
    return eigenvec
    


kvec = np.linspace(-1, 1, 2**6)
kvec = np.ndarray.tolist(kvec)
args = np.array([kvec, 1, 0])

def eigenmesh(func, i, *args):


    kx_vec = args[0]
    ky_vec = args[1]
        
    args = np.array(args)
    
    evals = np.zeros((len(kx_vec), len(ky_vec)))
    scalar_arg = args

    for j, kx in enumerate(kx_vec):
        for k, ky in enumerate(ky_vec):
                                
                scalar_arg[0] = kx
                scalar_arg[1] = ky
                evals[j, k] = eigen(func, i, *scalar_arg)
               
            
    return evals
    
def eigenvecmesh(func, i, *args):


    kx_vec = args[0]
    ky_vec = args[1]
        
    args = np.array(args)
    
    evecs = np.zeros((2, len(kx_vec), len(ky_vec)), dtype='complex')
    scalar_arg = args
#    evecs = []
    for j, kx in enumerate(kx_vec):
        for k, ky in enumerate(ky_vec):
                                
                scalar_arg[0] = kx
                scalar_arg[1] = ky
#                evecs[j, k] = eigenvec(func, i, *scalar_arg)
                evecs[0, j, k] = eigenvec(func, i, *scalar_arg)[0]
                evecs[1, j, k] = eigenvec(func, i, *scalar_arg)[1]
                
    
#    evecs = np.array([evecs])
#    evecs = evecs.reshape((2,len(kx_vec), len(ky_vec)))
                         
               
            
    return evecs
    
eigenarray = np.vectorize(eigen) 

#%%
kvec = np.linspace(-1*1e-0, 1*1e-0, 40)
plt.imshow(np.real(eigenmesh(H_Rashba, 0, *[kvec, kvec, 0])))

evecs = eigenvecmesh(H_Rashba, 0, *[kvec, kvec, 0*1.5])
plt.show()
plt.imshow(np.imag(evecs[0]))
#%%
nk = len(kvec)
evecs = evecs.reshape(nk, nk, 2)
plt.imshow(np.real(evecs[:,:,0]), interpolation='none')
plt.colorbar()

#%%
evec = []
kvec = np.linspace(-1, 1, 200)
for kx in kvec:
    for ky in kvec:
        
        vec = 1/np.sqrt(2) * np.array( [1.0 + 0.0j, -kx*1.0j-ky])
        evec.append(vec)

evec = np.array(evec)
nk = len(kvec)
evec = evec.reshape(len(kvec), len(kvec),2)
dotx = np.zeros((nk-1, nk-1), dtype=complex)
doty = np.zeros((nk-1, nk-1), dtype=complex)

def dx(array):
    
    dx = np.diff(array, axis=0)
    
    return dx
    
def dy(array):
    
    dy = np.diff(array, axis=1)
    
    return dy
    
for i in range(len(kvec)-1):
    for j in range(len(kvec)-1):
    
        dotx[i,j] = np.dot(evec[i,j,:], dx(evec)[i,j,:])
        doty[i,j] = np.dot(evec[i,j,:], dy(evec)[i,j,:])
        
curv = dx(doty)[:,0:-1] - dy(dotx)[0:-1,:]

plt.imshow(np.imag(curv), interpolation='none')

#%%

def Rashba_evec(kvec):
    
    nk = len(kvec)
    evec = np.zeros([2,nk, nk], dtype=complex)
    for i, kx in enumerate(kvec):
        for j, ky in enumerate(kvec):
            
            evec[0,i,j], evec[1,i,j] = 1/np.sqrt(2) * np.array( [1.0 + 0.0j, (-kx*1.0j-ky)/ np.sqrt(kx**2 + ky**2)])
            
    return evec
    
#def dx(vec):

evec = Rashba_evec(kvec)