import numpy as np
import h5py
import time
import coacs
import random
from tempfile import TemporaryFile

import scipy as sp

# todo: refactor into functions and perhaps class
# load complex reference matrix
f = h5py.File('reference.mat', 'r')
f2 = sp.io.loadmat('/home/noax/jackdaw/COACS/rpois.mat')
vars = list(f.keys())
reference = f['reference'][:]
r3b = f['r3b'][:]
r = f['r'][:]
mask = f['mask'][:]
reference = reference['real'] + reference['imag'] * 1j

test_sampling = f2['r']
test_r = test_sampling.transpose()

rounds = 68
# change to ndarrays?
qbarrier = []
nzpenalty = []
iters = []
tols = []

# todo: review this line
if 'nowindow' not in locals() and 'nowindow' not in globals():
    nowindow = []  # should this be a bool?

# prepare settings for the different continuation levels
for i in range(rounds):
    val = 2 ** (3 - i)
    qbarrier.append(val)
    nzpval = 1e4 / val
    nzpenalty.append(nzpval)
    iters.append(6e2)
    tolval = val * 1e-14
    tols.append(tolval)

numrep = 1
# cell arrays in matlab
rs = np.empty((numrep, 256, 256))
vs = np.empty((numrep, 256, 256))

random.seed(0)

for qq2 in range(numrep):
    banner = print("################## PREP REPLICATE ", qq2)
    r2 = np.random.poisson(r3b)
    r[r >= 0] = r2[r >= 0]
    rs[qq2] = r

#### TODO: parallelize
#### note: matlab uses parfor
for qq2 in range(numrep):
    banner = print("################## REPLICATE ", qq2)
    r = rs[qq2]
    tic = time.time()
    v, b = coacs.heal(test_r, mask, np.zeros((256, 256)), [], 'AT', len(qbarrier), qbarrier, nzpenalty, iters, tols, nowindow)

    # TODO: check correctness below
    toc = time.time()
    vs[qq2 - 1] = v

rsold = rs
vsold = vs

rs = np.empty(50)
vs = np.empty(50)

for qq2 in range(numrep):
    rs[qq2] = rsold[qq2]
    vs[qq2] = vsold[qq2]

print(vs)

np.save("pattern.npy", vs)

rsold = None
vsold = None

