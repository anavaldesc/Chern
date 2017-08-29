from __future__ import print_function, division
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import databandit as db
import pandas as pd
from fnmatch import fnmatch
import sys
sys.path.append('/Users/banano/databandit/databandit')


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


#with open('dataruns.yaml') as f:
#    b = db.ordered_load(f)
#
#
transition = 'zx_strong_raman'
redo_prepare = True


p0 = []
p1 = []
p2 = []
od = []
arpfinebias = []
delta_xyz = []

camera = 
date = 20170630
sequence = 45

folder = getfolder(date, sequence)
outfile = '{}_{}_{:04d}.h5'.format(transition, date, sequence)
                                 
try:
    with h5py.File('results/' + outfile, 'r') as f:
        f['data']
except KeyError:
    redo_prepare = True
except IOError:
    redo_prepare = True


if redo_prepare:
    print('Preparing {} data...'.format(transition))
    for r, file in matchfiles(folder):
        with h5py.File(os.path.join(r, file), 'r') as f:
            attrs = f['results/rois_od'].attrs
            print(attrs[:])

            try:
                p0.append(attrs['roi_0'])
                p1.append(attrs['roi_1'])
                p2.append(attrs['roi_2'])
                od.append(attrs['optdepth'])
            except KeyError:
                p0.append(np.nan)
                p1.append(np.nan)
                p2.append(np.nan)

            attrs = f['globals'].attrs
            arpfinebias.append(attrs['ARP_FineBias'])
            delta_xyz.append(attrs['delta_xyz'])

        # break

    df = pd.DataFrame()
    df['p0'] = p0
    df['p1'] = p1
    df['p2'] = p2
    df['arpfinebias'] = arpfinebias
    df['delta_xyz'] = delta_xyz

    df = df.dropna()
    df = df.sort_values(by='arpfinebias')
    df.to_hdf('results/' + outfile, 'data', mode='w')
else:
    df = pd.read_hdf('results/' + outfile, 'data')

# Analyse data

print('Analysing {} data...'.format(transition))
bias_list = df.arpfinebias.unique()

popt = []
perr = []
data_min = []

plotme = True

# Prime the fitting routine
if transition == 'zy':
    center = 395 + np.linspace(-3.3, 3.3, len(bias_list))**2
elif transition == 'xy':
    center = 165 + np.linspace(-3.3, 3.2, 20)**2
elif transition.startswith('zx'):
    center = [235] * len(bias_list)

for idx, bias in enumerate(bias_list):

    p0 = df.p0[np.isclose(df.arpfinebias, bias)]
    p1 = df.p1[np.isclose(df.arpfinebias, bias)]
    p2 = df.p2[np.isclose(df.arpfinebias, bias)]
    delta_xyz = df.delta_xyz[np.isclose(df.arpfinebias, bias)]

    if transition.startswith('zx'):
        p = p0 / (p0 + p1)
    if transition == 'zy':
        p = p0 / (p0 + p2)
    if transition == 'xy':
        p = p1 / (p1 + p2)

    # keep the min to compare with the fit
    data_min.append(p.argmin())

    try:
        # amplitude -0.2 for xy, otherwise -0.4 works
        popt0, perr0 = db.tr.fit_sinc(delta_xyz, p,
                                      pars_guess=(-0.2, center[idx], 7, 1))

        popt.append(popt0[1])
        perr.append(perr0[1])

        if plotme:
            delta_fit = np.linspace(delta_xyz.min(), delta_xyz.max(), 100)
            plt.plot(delta_fit, db.tr.sincfn(delta_fit, *popt0))
    except Exception as e:
        print(e)

    if plotme:
        plt.plot(delta_xyz, p, 'o', label=bias-40e-3)
        plt.legend()
        plt.ylim([0, 1])
        plt.show()


df = pd.DataFrame()
df['arpfinebias'] = bias_list
df['popt'] = popt
df['perr'] = perr
df['data_min'] = data_min
df.to_hdf('results/' + outfile, 'results', mode='a')
