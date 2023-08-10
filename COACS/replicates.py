import numpy as np
import h5py
import time
import coacs
import random

# todo: refactor into functions and perhaps class
# load complex reference matrix
f = h5py.File('reference.mat', 'r')
vars = list(f.keys())
reference = f['reference'][:]
r3b = f['r3b'][:]
r = f['r'][:]
f2 = f['f2'][:]
mask = f['mask'][:]
reference = reference['real'] + reference['imag'] * 1j

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
    val = 2 ** (2 - i)
    qbarrier.append(val)
    nzpval = 1e4 / val
    nzpenalty.append(nzpval)
    iters.append(6e2)
    tolval = val * 1e-14
    tols.append(tolval)

numrep = 50
# cell arrays in matlab
rs = np.empty((50, 256, 256))
vs = np.empty((50, 256, 256))

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
    r = rs[qq2 - 1]
    tic = time.time()
    v, b = coacs.Healer(r, mask, np.zeros((256, 256)), [], 'AT', len(qbarrier), qbarrier, nzpenalty, iters, tols, nowindow)

    # TODO: check correctness below
    toc = time.time()
    vs[qq2 - 1] = v

rsold = rs
vsold = vs

rs = np.empty(50)
vs = np.empty(50)

for qq2 in range(numrep):
    rs[qq2 - 1] = rsold[qq2 - 1]
    vs[qq2 - 1] = vsold[qq2 - 1]

rsold = None
vsold = None

