# Convex Optimization of Autocorrelation with Constrained Support (COACS)
# Author: Noa Lerch
# Based on code by Carl Nettelblad at https://github.com/cnettel/jackdaw
# As well as the TFOCS library for MATLAB: http://cvxr.com/tfocs/
# Bachelor's degree project in Computer Science at Uppsala University
# 2022

import numpy as np

class Healer:
    def __init__(self, pattern, support, bkg, init_guess, num_rounds, qbarrier,
            nzpenalty, iters, tolerance, nowindow):
        """The main COACS function
        pattern : ??
            pattern to phase
        support : ??
            support mask (in autocorrelation space
        bkg : ??
            background signal, support for non-zero values here basically stale
        init_guess : ??
            start guess for the pattern
        alg : ??
            solving algorithm to use
        num_rounds : int
            number of outermost iterations
        qbarrier : int?
            qbarrier (2 * l) in each round
        nzpenalty : ??
            penalty constant outside of the support
        iters : int
            number of TFOCS (solver) iterations within the round
        tolerance : ??
            the tolerance used to determine end of iteration in TFOCS
        nowindow: ??
        """

        # TODO: if nargin < 11

        iter_factor = 1.1

        # handle scalars
        nzpenalty = np.multiply(np.ones(1, num_rounds), nzpenalty)
        qbarrier = np.multiply(np.ones(1, num_rounds), qbarrier)

        # TODO: getdims
