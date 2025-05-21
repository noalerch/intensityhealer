# Convex Optimization of Autocorrelation with Constrained Support (COACS)
# Author: Noa Lerch
# Based on code by Carl Nettelblad at https://github.com/cnettel/jackdaw
# As well as the TFOCS library for MATLAB: http://cvxr.com/tfocs/
# Bachelor's degree project in Computer Science at Uppsala University
# 2023
#import scipy.fftpack as sfft
#import scipy as sp
import numpy as np
import coacsutils as cu
import sys
import h5py
#sys.path.append('../')
#sys.path.append('../')
sys.path.append('/home/stefon/phasehealer/')
#sys.path.append('/home/stefon/phasehealer/*')
print(sys.path)
from ConicSolvers import ConicSolver
#import ConicSolvers as cs


# class Healer:
    #def __init__(self, pattern, support, bkg, init_guess, alg, num_rounds, qbarrier,
    #             nzpenalty, iters, tolerance, nowindow=None):
def heal(pattern, support, bkg, init_guess, alg, num_rounds, qbarrier,
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
        penalty constant outside the support
    iters : int
        number of TFOCS (solver) iterations within the round
    tolerance : ??
        the tolerance used to determine end of iteration in TFOCS
    nowindow: ??
    """

    # TODO: if nargin < 11


    iter_factor = 1.1

    # handle scalars
    # 1x69 in matlab, is this correct?
    # resulting from num_rounds
    # TODO: check this, possible error in matlab
    # nzpenalty = np.ones(num_rounds) * nzpenalty  # redundant if nzpenalty is 1D
    qbarrier = np.ones(num_rounds) * qbarrier

    dims, side2, fullsize, pshape, cshape = cu.get_dims(pattern)
    pshape = [int(i) for i in pshape]
    cshape = [int(i) for i in cshape]

    original_pattern = pattern.copy()
    pattern = pattern.reshape(fullsize, 1).flatten()


    mask = np.concatenate([np.reshape(support, pshape), np.zeros(np.reshape(support, pshape).shape)])
    mask = np.reshape(mask, (fullsize * 2, ))

    factor = []
    if nowindow:
        factor = np.ones(65536)
        base_penalty = 1 - mask
    else:
        #factor, base_penalty = cu.create_windows(original_pattern, mask, qbarrier[0], np.ones(fullsize))
        factor, base_penalty = cu.create_windows(original_pattern, mask)
        #factor = np.asarray(factor)
        base_penalty = np.asarray(base_penalty)

    if len(nowindow) == 0:
        nowindow = False

    # purely ie zero mask in imaginary space
    our_linp = jackdaw_linop(original_pattern, 1)
    # no windowing used within linop for now
    #our_linp = our_linp_flat

    if not init_guess:
        print(type(pattern))
        print(type(factor))
        init_guess = pattern * factor
        # replace neg values with 0
        init_guess[init_guess < 0] = 0

    # move to device
    x = np.asarray(init_guess.flatten())
    x_prev = x.copy()
    y = x.copy()
    jval = 0

    filter = np.ones_like(factor)
    rfilter = filter.copy()

    # i is outer round
    for i in range(num_rounds):

        print("outer round: ", i)
        # looks ok
        penalty = base_penalty * nzpenalty[i]

        # acceleration scheme based on assumption of linear steps
        # in response to decreasing qbarrier
        if i > 0 and qbarrier[i] != qbarrier[i - 1]:
            if jval > 0:
                diffx = x + (x - x_prev) * (qbarrier[i] / qbarrier[i - 1])
                smoothop = cu.diffpoisson(factor, pattern, diffx.flatten(), bkg.flatten(), diffx, filter, qbarrier[i])
                op, diffxt, level, xlevel = cu.create_proxop(diffx, penalty, our_linp)

                f = lambda z: smoothop(z + x - diffx) + proxop(our_linp(z + x - diffx, 2))
                y = x + half_bounded_line_search(x - x_prev, f)

            step = np.linalg.norm(y - x)
            x_prev = x.copy()

            jval = jval + 1

        x_prev_inner = y.copy()
        j_val_inner = -1


        #solver.max_iterations = int(np.ceil(iters[i] / iter_factor))
        max_iters = int(np.ceil(iters[i] / iter_factor))

        # inner acceleration scheme
        # based on overall difference to previous pre-acceleration start
        while True:
            if j_val_inner >= 0:
                diffx = y.copy()

                smoothop = cu.diffpoisson(factor, pattern, diffx.flatten(), bkg.flatten(), diffx, filter, qbarrier[i])

                proxop, diffxt, level, xlevel = cu.create_proxop(diffx, penalty, our_linp)

                f_3 = lambda z: smoothop(np.asarray(z)) + proxop(our_linp(z - xlevel, 2))

                x = y + half_bounded_line_search(y - x_prev_inner, f_3)
            else:
                x = y.copy()

            x_prev_inner = y.copy()
            j_val_inner += 1


            # is this correct?
            diffx = np.copy(x)

            smoothop = cu.diffpoisson(factor, pattern, diffx.flatten(), np.array(bkg.flatten()), diffx, filter, qbarrier[i])


            proxop, diffxt, level, xlevel = cu.create_proxop(diffx, penalty, our_linp)

            solver = ConicSolver.ConicSolver()
            solver.alg = alg
            solver.restart = 5e5
            solver.count_ops = True
            solver.print_stop_criterion = True
            solver.print_every = 2000
            # no regress restart option
            solver.restart = -100000

            solver.autoRestart = 'fun, gra'

            max_iters = np.ceil(max_iters * iter_factor)
            solver.max_iterations = max_iters
            solver.tolerance = tolerance[i]
            solver.L_0 = 2 / qbarrier[i]
            solver.L_exact = solver.L_0
            #solver.alpha = 0.1
            solver.beta = 1

            # x shockingly close after first iteration
            x, out = solver.solve(smoothop, our_linp, proxop, -level, affine_offset=xlevel)

            xt_update = x.copy()
            x = our_linp(x, 1)
            x_step = x.copy()
            x_update = x + xlevel
            old_y = y.copy()
            prev_step = x_prev_inner - diffx
            level_prev_diff = np.linalg.norm(prev_step)
            y = x_update.flatten() + diffx.flatten()

            # flatten necessary?
            f_4 = lambda x: smoothop(x + x_update.flatten()) + proxop(our_linp(x + x_step, 2))
            y += half_bounded_line_search(x_update, f_4)  # pretty good after 1 iter

            level_x_diff = np.linalg.norm(x_prev_inner - y)

            # hopefully non-issue
            if level_prev_diff > level_x_diff:
                print("Do we need to reset acceleration?")

            x = y.copy()

            #global x2

            x2 = x.copy()
            np.savez('output_file.npz', x3="x3", x2="x2")

            # here we use matlab syntax
            # save x26 x2
            # we should investigate this further

            diffstep = x_prev_inner - y

            positive_pattern = pattern[pattern >= 0]
            positive_factor = factor[pattern >= 0]

            # rchange sliiightly off but maybe is ok
            rchange = np.linalg.norm(diffstep[pattern >= 0], ord=1) / np.linalg.norm(
                positive_pattern * positive_factor, ord=1)
            print(j_val_inner)

            if j_val_inner >= 0 and rchange < 1e-6 * out.n_iter and \
                abs(smoothop(y - diffx) - smoothop(x_prev_inner - diffx) + proxop(our_linp(y - diffx + xlevel, 2)) \
                    - proxop(our_linp(x_prev_inner, 2) - diffxt - level)) < 1e-4 * out.n_iter:
                print("Next outer iteration")
                break

            if out.n_iter < solver.max_iterations:
                print("Reverting max_iterations")
                solver.max_iterations = np.ceil(solver.max_iterations / iter_factor)

    out_pattern = np.reshape(x, pshape)

    return out_pattern, out # , factor





# TODO: test this
def jackdaw_linop(pattern, filter):
    dims, side2, fullsize, pshape, cshape = cu.get_dims(pattern)

    if dims == 3:  # code not reached in coacs
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
            to_shift = x[0:side, :] + 1j * x[side:2*side, :]
            # This looks correct!
            x = np.fft.fftshift(to_shift)

        x2 = x * np.conj(shifter)

        # looks sorta ok
        x = np.fft.fftn(x2) * np.conj(shifter)

        y = (side ** (-dims / 2)) * np.real(x.flatten()) * filter  # might not work if filter is an array
    elif mode == 2:  # 23-11-22 verified for first iteration
        x2 = np.zeros(pshape)
        x2 = np.real(x) * filter
        x2 = x2.reshape(pshape)
        x2 = x2 * shifter

        # i think this is where it gets weird
        #f2 = sp.io.loadmat('/home/noax/jackdaw/COACS/x2.mat')
        x2 = np.fft.ifftn(x2)
        #asum = sum(sum(x2))
        #num_diff = x2 - f2['x2'].transpose() # numerical differences between matlab and python: very small

        # small numerical differences
        x2 = x2 * shifter
        x2 = np.fft.ifftshift(x2)

        x3 = (side ** (dims / 2)) * x2
        real_part = np.real(x3)
        imag_part = np.imag(x3)
        # np or np.
        y = np.concatenate((real_part.ravel(), imag_part.ravel()))
        y = y.reshape((2 * fullsize, 1))

        # tiny numerical error
        y = y.flatten()

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
        #idx = np.argmin(real_vals)
        idx = real_vals.index(min(real_vals))

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




