import cupy as cp
import numpy as np
import h5py
import os
import time
import coacs
import random
from scipy import io

# todo: refactor into functions and perhaps class
# load complex reference matrix
f = h5py.File('reference.mat', 'r')
f2 = io.loadmat('pois2.mat')
vars = list(f.keys())
reference = f['reference'][:]
r3b = f['r3b'][:]
r_np = f['r'][:]

# convert r_np to cupy array
r = cp.array(r_np)

mask = cp.asarray(f['mask'][:])
reference = reference['real'] + reference['imag'] * 1j

# load pattern
pat = np.load('pattern.npy')
pat2 = np.load('pattern2.npy')

# test sampling taken from original matlab code.
test_sampling = f2['r']
np.save('../img/test.npy', test_sampling)

# use this for a more deterministic r
test_r = cp.asarray(test_sampling.transpose())

rounds = 2
# change to ndarrays?
qbarrier = np.empty(rounds)
nzpenalty = np.empty(rounds)
iters = np.empty(rounds)
tols = np.empty(rounds)

# todo: review this line
if 'nowindow' not in locals() and 'nowindow' not in globals():
    nowindow = []  # should this be a bool?

# prepare settings for the different continuation levels
for i in range(rounds):
    val = 2 ** (3 - i)
    qbarrier[i] = val
    nzpval = 1e4 / val
    nzpenalty[i] = nzpval
    iters[i] = 6e2
    tolval = val * 1e-14
    tols[i] = tolval

numrep = 2
# cell arrays in matlab
rs = cp.empty((numrep, 256, 256))
vs = cp.empty((numrep, 256, 256))

random.seed(0)

cp.cuda.runtime.profilerStart()
for qq2 in range(numrep):
    banner = print("################## PREP REPLICATE ", qq2)
    r2 = cp.array(np.random.poisson(r3b))
    print(type(r))
    print(type(r2))
    r[cp.where(r >= 0)] = r2[cp.where(r >= 0)]
    rs[qq2] = cp.array(r)
cp.cuda.runtime.profilerStop()

# save patterns to a dir
suffix = str(rounds) + str(numrep)
path = f"../img/pattern{suffix}"
if not (os.path.exists(path) and os.path.isdir(path)):
    os.mkdir(path)
    os.mkdir(f"{path}/rs")
    os.mkdir(f"{path}/vs")
for qq2 in range(numrep):
    banner = print("################## REPLICATE ", qq2)
    r = rs[qq2]
    tic = time.time()
    v, b = coacs.heal(r, mask, np.zeros((256, 256)), [], 'AT', len(qbarrier), qbarrier, nzpenalty, iters, tols, nowindow)
    print("## Finished heal ##")

    # TODO: check correctness below
    toc = time.time()
    with open(f"{path}/{suffix}time.txt", "w") as file:
        print(f"Replicate {qq2 + 1} Time: {toc-tic:}\n", file=file)
    vs[qq2 - 1] = v
    np.save(f"{path}/vs/rep{qq2 + 1}.npy", v)
    print("Time to finish heal: (seconds)")
    print(toc - tic)



rsold = rs
vsold = vs

rs = np.empty(numrep)
vs = np.empty(numrep)

#for qq2 in range(numrep):
#    rs[qq2] = rsold[qq2]
#    vs[qq2] = vsold[qq2]


np.save(f"../img/patternrs{suffix}.npy", rsold)
np.save(f"../img/patternvs{suffix}.npy", vsold)
np.save("itworked5050.npy", vsold)

rsold = None
vsold = None

