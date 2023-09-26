# Convex Optimization of Autocorrelation with Constrained Support (COACS)
# Author: Noa Lerch
# Based on code by Carl Nettelblad at https://github.com/cnettel/jackdaw
# As well as the TFOCS library for MATLAB: http://cvxr.com/tfocs/
# Bachelor's degree project in Computer Science at Uppsala University
# 2023

import numpy as np
import coacsutils as cu
import sys

sys.path.append('ConicSolver')
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

        if len(nowindow) == 0:
            nowindow = False

        iter_factor = 1.1

        # handle scalars
        # 1x69 in matlab, is this correct?
        # resulting from num_rounds
        # TODO: check this, possible error in matlab
        # nzpenalty = np.ones(num_rounds) * nzpenalty  # redundant if nzpenalty is 1D
        qbarrier = np.ones(num_rounds) * qbarrier

        dims, side2, fullsize, pshape, cshape = cu.get_dims(pattern)

        original_pattern = pattern.copy()
        # TODO: check dimensions of pattern
        pattern = pattern.reshape(fullsize, 1)

        solver = cs.ConicSolver()
        solver.alg = alg
        solver.restart = 5e5
        solver.count_ops = True
        solver.print_stop_criterion = True
        solver.print_every = 2500
        # no regress restart option
        solver.restart = -10000000  # ?

        solver.autoRestart = 'fun' # ? double check this

        mask = np.hstack((np.reshape(support, pshape), np.zeros(np.reshape(support, pshape).shape)))
        # TODO: check dimensions of mask
        mask = np.reshape(mask, (fullsize * 2, 1))

        # purely ie zero mask in imaginary space
        our_linp_flat = jackdaw_linop(original_pattern, 1)
        # no windowing used within linop for now
        our_linp = our_linp_flat

        # global factor

        # empty guess?
        if not init_guess:
            init_guess = pattern.flatten()
            # replace neg values with 0
            init_guess[init_guess < 0] = 0

        x = np.reshape(init_guess, (1, fullsize))
        # need to copy to not overwrite later
        x_prev = x.copy()
        y = x.copy()
        jval = 0

        filter = np.ones((fullsize, 1))
        rfilter = 1.0 / filter

        # i is outer round
        for i in range(num_rounds):
            # base_penalty = None
            if nowindow:
                factor = np.ones((65536, 1))
                base_penalty = 1 - mask
            else:
                factor, base_penalty = cu.create_windows(original_pattern, mask, qbarrier[i], filter)

            y = y * factor
            x = x * factor

            penalty = base_penalty * nzpenalty[i]

            # acceleration scheme based on assumption of linear steps
            # in response to decreasing qbarrier
            if i > 0 and qbarrier[i] != qbarrier[i - 1]:
                x_prev = x_prev * factor
                if jval > 0:
                    diffx = x + (x - x_prev) * (qbarrier[i] / qbarrier[i - 1])
                    smoothop = cu.diffpoisson(factor, pattern.flatten(), diffx.flatten(), bkg.flatten(), diffx, filter, qbarrier[i])

            x_prev_inner = y.copy()
            j_val_inner = -1  # ? might depend on what this is used for
            solver.maxIter = np.ceil(iters[i] / iter_factor)

            # inner acceleration scheme
            # based on overall difference to previous pre-acceleration start
            while True:
                if j_val_inner >= 0:
                    diffx = y.copy()

                    smoothop = cu.diffpoisson(factor, pattern.flatten(), diffx.flatten(), bkg.flatten(), diffx, filter, qbarrier[i])

                    proxop, diffxt, level, xlevel = cu.create_proxop(diffx, penalty, our_linp)

                    newstep = our_linp(x - x_prev, 2)
                    negstep = -newstep
                    negstep[penalty == 0] = 0
                    newstep[penalty > 0] = 0
                    negstep = our_linp(negstep, 1)
                    newstep = our_linp(newstep, 1)

                    ############ ignore this for now
                    y = x + half_bounded_line_search()

                    x = y + half_bounded_line_search() # ooga booga
                else:
                    x = y.copy()

                x_prev_inner = y.copy()
                j_val_inner = j_val_inner + 1
                solver.max_iterations = np.ceil(solver.max_iterations * iter_factor)
                solver.tolerance = tolerance[i]
                solver.L_0 = 2 / qbarrier[i]
                solver.L_exact = solver.L_0 * (96*96*96)**0.5
                solver.alpha = 0.1
                solver.beta = 0.1
                diffx = x.copy()

                # is it even necessary to flatten?
                #smoothop = cu.diffpoisson(factor, pattern.flatten(), diffx.flatten(), bkg.flatten(), diffx, filter, qbarrier[i])
                smoothop = cu.diffpoisson(factor, pattern, diffx, bkg, diffx, filter, qbarrier[i])
                proxop, diffxt, level, xlevel = cu.create_proxop(diffx, penalty, our_linp)

                # TODO: verify that solver attributes are correct

                # TODO: fix affine function
                # we get affine = {our_linp, xlevel}
                # our_linp is a function, xlevel is an array, probably 65536x1
                # this i believe represents the offset of the affine function

                x, out = solver.solve(smoothop, our_linp, proxop, -xlevel, affine_offset=xlevel)



# TODO: test this
def jackdaw_linop(pattern, filter):
    dims, side2, fullsize, pshape, cshape = cu.get_dims(pattern)

    if dims == 3:
        r = np.fftshift(np.pi / 2 + (np.arange(0.25, side2 - 0.75) * np.pi / side2))
        Xs, Ys, Zs = np.meshgrid(r, r, r)
        shifter = np.exp(1j * (Xs + Ys + Zs))

        r = np.linspace(0, -np.pi + np.pi / side2, side2)
        Xs, Ys, Zs = np.meshgrid(r, r, r)
        unshifter = np.exp(1j * (Xs + Ys + Zs))
    else:
        shifter = 1
        unshifter = 1

    linop = lambda x, mode: linop_helper(x, mode, dims, side2, fullsize, pshape, cshape, filter, unshifter, shifter)
    return linop


def linop_helper(x, mode, dims, side, fullsize, pshape, cshape, filter, unshifter, shifter):
    y = None
    if mode == 0:
        y = np.array([fullsize, 2 * fullsize])
    elif mode == 1:
        x = x.reshape(cshape)
        if dims == 3:
            x = np.fft.fftshift(x[0:side, 0:side, 0:side] + 1j * x[0:side, side:side * 2, 0:side])
        else:
            #x = np.fft.fftshift(x[side:side*2, :side] + 1j * x[:side, :side])
            # Split the array into two parts and add the imaginary part
            #part1 = x[:side, :side] + 1j * x[side:side*2, :side]
            #part2 = x[side:side * 2, :side]
            part1 = x[:side, :side]
            part2 = 1j * x[side:side*2, :side]

            # Perform the fftshift-like operation by swapping parts
            x = (part2 + part1)
            x = np.fft.fftshift(x)


            print(np.min(x))
            # NOTE: fftshift is not the same as fftshift in matlab
            # the matlab version does not shift the 0-frequency component

        x2 = x.copy() * np.conj(shifter)
        x = np.fft.fftn(x2) * np.conj(shifter)

        y = (side ** (-dims / 2)) * np.real(x.flatten()) * filter  # might not work if filter is an array
    elif mode == 2:
        # x2 = np.zeros(pshape)
        #x2 = np.real(x) * filter
        #x2 = x2 * shifter
        #x2 = np.fft.ifftn(x2)
        #x2 = x2 * shifter
        #x2 = np.fft.ifftshift(x2)
        #x2 = np.concatenate([np.real(side ** (dims / 2) * x2), np.imag(side ** (dims / 2) * x2)], axis=0)
#       # y = np.concatenate([np.real((side ** (dims / 2)) * x2), np.imag(side ** (dims / 2) * x2)]).reshape(1, 2 * fullsize)
        #real_part = np.real(side ** (dims / 2) * x2)
        #imag_part = np.imag(side ** (dims / 2) * x2)

    # Combine the real and imaginary parts
        #y = np.concatenate([real_part, imag_part]).reshape(1, 2 * fullsize)
#        y = np.reshape(x2, (2 * fullsize))

        ####################
        ####################
        ####################
        ## SUSPICOUS RESULTS
        x2 = np.zeros(pshape)
        x2 = (x.real * filter).reshape(pshape)

        x2 = x2 * shifter
        # different results from matlab
        x2 = np.fft.ifftn(x2)
        # transpose?
        x2 = x2 * shifter
        x2 = np.fft.ifftshift(x2)
        test = x2.flatten()

        # we want y to be between about -10 and 33
        y = np.concatenate([x2.real, x2.imag]).reshape((2 * fullsize, 1))
        y = y.flatten()

    assert y is not None
    return y

def half_bounded_line_search(y, f):
    factor = 1.0  # do check if factor should actually be global
    last_val = f(y * factor)
    min_overall = 0.0
    min_val = f(min_overall)

    while True:
        factor = factor * 2
        new_val = f(y * factor)
        if new_val >= last_val:
            break
        if new_val < min_val:
            min_overall = factor
            min_val = new_val

    lo = 0.0
    hi = factor

    for i in range(100):
        diff = hi - lo
        poses = (lo + diff / 3, lo + 2 * diff / 3)
        real_vals = (f(y * poses[0]), f(y * poses[1]))
        # a, b = min(real_vals)
        idx = np.argmin(real_vals)

        if real_vals[1] == real_vals[0]:
            break

        if idx == 0:
            hi = poses[1]
        elif idx == 1:
            lo = poses[0]

        if real_vals[idx] < min_val:
            min_overall = poses[idx]
            min_val = real_vals[idx]

        if min_overall < lo:
            lo = min_overall
            break

        hi = max(min_overall, hi)
        last_val = real_vals[idx]  # redundant?
    x = y * lo

    return x




