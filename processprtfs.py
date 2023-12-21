#!/usr/bin/env python

import sys, h5py
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
import spimage
import argparse

parser = argparse.ArgumentParser(prog='processprtfs.py', description='Compute superimposed image and phase-retrieval transfer functions.')
parser.add_argument('-a', '--phase',  metavar='PHASE', type=str, default='vs72/phasing.h5', help='Phasing results for patterns')
parser.add_argument('-r', '--ref',  metavar='REF', type=str, default='reference.mat', help='File containing reference pattern')
args = parser.parse_args()


f = h5py.File(args.phase, 'r+')

recons  = f['real_space_final'][:]
#fourier = f['fourier_space_final'][:]
support = f['support_final'][:]
rerror  = f['real_error'][:]
#ferror  = f['fourier_error'][:]

with h5py.File(args.ref, 'r') as reff:
    f2 = np.reshape(reff['f2'][:],(256,256))
    reference = reff['reference'][:]
    reference = np.fft.ifftshift(np.fft.fft2(np.fft.fft2(np.fft.fftshift(reference['real'] + 1j * reference['imag'])) * np.sqrt(f2)))

# Loop over 50 original images
M = 50
N = 100
subsetcount = 10

allsupers = np.zeros((M, 256, 256), dtype=np.complex128)

for m in range(M):
    indices = np.argsort(rerror[(m*N):((m+1)*N)]) + m * N
    indices = indices[0:subsetcount]
    print(indices)
    output_prtf = spimage.prtf(recons[indices], support[indices], enantio=True, translate=True, clearsupport=True, reference=reference)
    allsupers[m,:,:] = output_prtf['super_image'] / subsetcount

try:
    del f['super_images']
except:
    pass

f['super_images'] = allsupers

