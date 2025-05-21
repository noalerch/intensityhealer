#!/bin/bash

# Run coacs
matlab -nodesktop -nosplash -r "replicates; save invicosa72orig -v7.3"

# Run coacs without apodization
#matlab -nodesktop -nosplash -r "nowindow = 1; replicates; save invicosa72nw -v7.3"

# Phase coacsed patterns
mkdir vs72
python phasing.py --output vs72/ --variable vs --filename invicosa72orig.mat

#mkdir vs72
#python phasing.py --output vs72nw/ --variable vs --filename invicosa72nw.mat

# Phase original ("rs") patterns
# Apply mask for the negative values, apply Hann window (in f2)
mkdir rs72f2
python phasing.py --output rs72f2/ --variable rs --f2 --mask --filename invicosa72orig.mat

mkdir rs72
python phasing.py --output rs72/ --variable rs --mask --filename invicosa72orig.mat

mkdir rs72f2sh
python phasingsh.py --output rs72f2sh/ --variable rs --f2 --mask --filename invicosa72orig.mat

python processprtfs.py --phase vs72/phasing.h5 --ref reference.mat
#python processprtfs.py --phase vs72nw/phasing.h5 --ref reference.mat
python processprtfs.py --phase rs72f2/phasing.h5 --ref reference.mat
python processprtfs.py --phase rs72f2sh/phasing.h5 --ref reference.mat
python processprtfs.py --phase rs72/phasing.h5 --ref reference.mat

# Do mosaic and compute R factors and MSE
python domosaic.py

# Create more manageable files
matlab -nojvm -nodesktop -nosplash -r "createsupers"

cp ossvs72/phasing.h5 ossvs72/phasingstrip.h5
python strip.py ossvs72/phasingstrip.h5
h5repack -f GZIP=9 ossvs72/phasingstrip.h5 ossvs72short.h5
rm ossvs72/phasingstrip.h5

cp ossrs72f2/phasing.h5 ossrs72f2/phasingstrip.h5
python strip.py ossrs72f2/phasingstrip.h5
h5repack -f GZIP=9 ossrs72f2/phasingstrip.h5 ossrs72f2short.h5
rm ossrs72f2/phasingstrip.h5

cp vs72/phasing.h5 vs72/phasingstrip.h5
python strip.py vs72/phasingstrip.h5
h5repack -f GZIP=9 vs72/phasingstrip.h5 vs72short.h5
rm vs72/phasingstrip.h5

cp vs72nw/phasing.h5 vs72nw/phasingstrip.h5
python strip.py vs72nw/phasingstrip.h5
h5repack -f GZIP=9 vs72nw/phasingstrip.h5 vs72nwshort.h5
rm vs72nw/phasingstrip.h5

cp rs72f2/phasing.h5 rs72f2/phasingstrip.h5
python strip.py rs72f2/phasingstrip.h5
h5repack -f GZIP=9 rs72f2/phasingstrip.h5 rs72f2short.h5
rm rs72f2/phasingstrip.h5

cp rs72f2sh/phasing.h5 rs72f2sh/phasingstrip.h5
python strip.py rs72f2sh/phasingstrip.h5
h5repack -f GZIP=9 rs72f2sh/phasingstrip.h5 rs72f2shshort.h5
rm rs72f2sh/phasingstrip.h5

cp rs72/phasing.h5 rs72/phasingstrip.h5
python strip.py rs72/phasingstrip.h5
h5repack -f GZIP=9 rs72/phasingstrip.h5 rs72short.h5
rm rs72/phasingstrip.h5

