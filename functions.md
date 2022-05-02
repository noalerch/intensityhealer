# TFOCS functions & scripts to be analyzed & rewritten


## jackdaw

### coacsemc.m
* healernoninv

### createproxop.m
args:
    diffx
    penalty
    ourlinp <- function with 2 args

* zero_tolerant_quad <-- jackdaw

### createsupers.m
* h5read (h5py)

### createwindows.m
args: 
    pattern
    mask
    qbarrier

* getdims() <-- jackdaw
* createfilter() <-- function declared inside createwindows.m
* hann() <-- signal library
* reshape() <-- matlab
* repmat <-- matlab, repeat copies of array
* fftshift <-- matlab, shift zero frequency of fourier transform to center

### diffpoisson.m

### getdims.m
* numel() <-- matlab, number of array elements

### halfboundedlinesearch.m

### healernoninv.m
args:
    pattern
    support
    bkg
    initguess
    alg
    numrounds
    qbarrier
    nzpenalty
    iters
    tols

* getdims() <-- jackdaw
* jackdawlinop()
* halfboundedlinesearch()
* diffpoisson

* tfocs() <-- main TFOCS function. First called without arguments for default return
calls tfocs, changes options for
- alg = alg (argument)
- maxmin = 1
- restart = 5e5
- countOps = 1
- printStopCrit = 1
- printEvery = 2500
- restart = -10000000
- autoRestart = fun

- maxIts, tol , L0, Lexact, alpha, beta = changed in forloop

line 174: tfocs({smoothop}, {outlinp,xlevel}, proxop, -level * 1, opts);


### jackdawlinop.m
conj() <-- matlab, compex conjugation

* features new function FFT2 helper() for TFOCS

### replicates.m (script)
* rng()

* parfor <-- parallel for-loop

* healernoninv()
* clear


## TFOCS: 
* tfocs
* 

## Other:
norm (euclidean norm of vector)
qbarrier() (passed as argument)
createwindows()


#### tfocs()

## tfocs()

* rmfield <-- matlab, remove field from structure

