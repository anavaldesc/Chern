# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:23:49 2017

@author: banano
"""

from collections import deque
from scipy.linalg import pinv, lu, solve
import numpy as np
import pandas as pd
import os
from fnmatch import fnmatch

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