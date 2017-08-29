# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:02:49 2017

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 50 * np.pi, )
omega = 200*0
gridsize = 1e3
phi = np.linspace(-3*np.pi, 3 * np.pi, gridsize)
fringes = []
k = 0.8
kvec = np.linspace(-k, k, gridsize)

for i in range(len(t)):
    y = np.sin((omega + kvec + 0*kvec) * t[i] + 0*phi*k)
    fringes.append(y)
#    plt.plot(y)
    
plt.show()
plt.pcolormesh(fringes, cmap='Greys')
plt.xlabel('q')
plt.ylabel('time')
plt.axis('Tight')