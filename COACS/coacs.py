# Convex Optimization of Autocorrelation with Constrained Support (COACS)
# Author: Noa Lerch
# Based on code by Carl Nettelblad at https://github.com/cnettel/jackdaw
# As well as the TFOCS library for MATLAB: http://cvxr.com/tfocs/
# Bachelor's degree project in Computer Science at Uppsala University
# 2023

import numpy as np
import coacsutils as cu
import ConicSolver as cs


class Healer:
    def __init__(self, pattern, support, bkg, init_guess, alg, num_rounds, qbarrier,
                 nzpenalty, iters, tolerance, nowindow=None):
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

        if nowindow is None:
            nowindow = []

        iter_factor = 1.1

        # handle scalars
        nzpenalty = np.multiply(np.ones(1, num_rounds), nzpenalty)
        qbarrier = np.multiply(np.ones(1, num_rounds), qbarrier)

        dims, side2, fullsize, pshape, cshape = cu.get_dims(pattern)

        original_pattern = pattern
        pattern = pattern.reshape(fullsize, 1)

        solver = cs.ConicSolver
        solver.alg = alg
        solver.restart = 5e5
        solver.count_ops = True
        solver.print_stop_criterion = True
        solver.print_every = 2500
        # no regress restart option
        solver.restart = -10000000  # ?

        solver.autoRestart = 'fun' # ? double check this

        mask = np.hstack((np.reshape(support, pshape), np.zeros(np.reshape(support, pshape).shape)))
        mask = np.reshape(mask, (fullsize * 2, 1))

        # purely ie zero mask in imaginary space
        our_linp_flat = jackdaw_linop(original_pattern, 1)
        # no windowing used within linop for now
        our_linp = our_linp_flat


        jackdaw_linop


