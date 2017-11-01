#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:08:28 2017

@author: banano
"""


import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit


def gaussian2D(xyVals, bkg, amp, x0, sigmax, y0, sigmay) :
    # A 2D gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background   : p[1]
    #   X Central value                : p[2]
    #   X Standard deviation           : p[3]
    #   Y Central value                : p[4]
    #   Y Standard deviation           : p[5]
    
    gauss2D = bkg + amp*np.exp(-1*(xyVals[0]-x0)**2/(2*sigmax**2)
                                -1*(xyVals[1]-y0)**2/(2*sigmay**2))
    
    return gauss2D

def grid2D(xyVals, p):
    
    # Frequency along x axis: p[0]
    # Frequency along y axis: p[1]
    # Phase along x axis: p[2]
    # Phase along y axis: p[3]
    d_x = xyVals[0].max() - xyVals[0].min()
    d_y = xyVals[1].max() - xyVals[1].min()
    
    grid = np.sin(p[0] * xyVals[0] / d_x + p[2] / d_x) ** 2
    grid +=  np.sin(p[1] * xyVals[1] / d_y + p[3] / d_y) ** 2
    grid = grid / grid.max()
    
    return grid
    
def roi_sum(img, x1, y1, x2, y2, w):
    
    from math import floor
    
    # x_1 value
    # x_2 value
    w = floor(w)
    x1 = floor(x1)
    y1 = floor(y1)
    x2 = floor(x2)
    y2 = floor(y2)
    
    roi1 = img[y1 - w:y1+w, x1-w: x1+w]
    roi2 = img[y2 - w:y2+w, x2-w: x2+w]
    roi_sum = roi1 + roi2
    
    return roi_sum


def fcn2min(params, xy_vals, img):
 
    x1 = params['x1']
    y1 = params['y1']
    x2 = params['x2']
    y2 = params['y2']
    w = params['w']
    bkg = params['bkg']
    amp = params['amp']
    x0 = params['x0']
    sigmax = params['sigmax']
    y0 = params['y0']
    sigmay = params['sigmay']    
    data = roi_sum(img, x1, y1, x2, y2, w)
    model = gaussian2D(xy_vals, bkg, amp, x0, sigmax, y0, sigmay)
#    
    return model - data

x1 = 2**9
x2 = x1
y1 = int(0.75 * 2**10)
y2 = int(0.25 * 2**10)
w = 120
gauss_params_1 = [0, 2, x1, 100, y1, 100]
gauss_params_2 = [0, 2, x2, 100, y2, 100]
grid_params = [20.0, 00, -2**9, 0]
x = np.arange(0, 2**10)
xy_vals = np.meshgrid(x, x)

grid2d = grid2D(xy_vals, grid_params)
gauss2d_1 = gaussian2D(xy_vals, *gauss_params_1)* grid2d
gauss2d_2 = gaussian2D(xy_vals, *gauss_params_2) * (1 - grid2d)
plt.imshow(gauss2d_1 + gauss2d_2)
plt.show()

y1 = y1 + 1
#%%
#Test lmffit

#y1 = y1+5
params = Parameters()
params.add('x1',   value= x1, vary=False)
params.add('y1', value= y1, vary=False)
params.add('x2',   value= x2+6,  min=x2-10, max=x2+10, brute_step=1)
params.add('y2', value= y2-5, min=y2-10, max=y2+10, brute_step=1)
params.add('w', value= w, vary=False)
params.add('bkg', value= 0.0, vary=False)# min=-0.1, max=0.1)
params.add('amp', value=2, vary=False)#min=1.5, max=2.5)
params.add('x0', value=150, vary=False)#min=130, max=170)
params.add('sigmax', value=100, vary=False)#min=50, max=150)
params.add('y0', value=150, vary=False)#min=130, max=170)
params.add('sigmay', value=100,  vary=False)#min=50, max=150)

img = gauss2d_1 + gauss2d_2
img = img * (1 + 0.5*1e-1 * np.random.rand(2**10, 2**10))

x_crop = np.arange(0, 2 * w)
xy_vals_crop = np.meshgrid(x_crop, x_crop)
plt.imshow(roi_sum(img, x1, y1, x2+6, y2-5, w))
#plt.imshow(fcn2min(params, xy_vals_crop, img))
plt.colorbar()
plt.show()
#plt.imshow(roi_sum(img, x1, y1, x2, y2, w))
plt.imshow(fcn2min(params, xy_vals_crop, img))
plt.colorbar()
plt.show()
minner = Minimizer(fcn2min, params, fcn_args=(xy_vals_crop, img))
result = minner.minimize('powell')
#'powell', 'cobyla', 'differential_evolution'
fit_params = []

for key in result.params.keys():
    
    fit_params.append(result.params[key].value)

reconstructed_img = roi_sum(img, *fit_params[0:5])
plt.imshow(reconstructed_img)
plt.colorbar()
plt.show()

# calculate final result
final = reconstructed_img + result.residual.reshape(2 * w, 2 * w)
plt.imshow(result.residual.reshape(2 * w, 2 * w), interpolation='none')
plt.colorbar()
plt.show()
## write error report
report_fit(result)