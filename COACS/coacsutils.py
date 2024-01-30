import numpy as np
import warnings

def get_dims(pattern):
    side2 = pattern.shape[0]
    fullsize = pattern.size

    if fullsize == side2 ** 3:
        dims = 3
        pshape = np.array([side2, side2, side2])
        cshape = np.array([side2, side2 * 2, side2])
    elif fullsize == side2 ** 2:
        dims = 2
        pshape = np.array([side2, side2])
        cshape = np.array([2 * side2, side2]) # opposite of matlab due to row-major matrices in numpy
    else:
        raise ValueError('Unknown dimensionality')

    return dims, side2, fullsize, pshape, cshape


def create_filter(filter, pshape: np.ndarray, side2, fullsize, dims):
    shape1 = pshape.copy()
    shape1[1] = 1 # opposite of matlab due to row-major matrices in numpy

    filter1 = np.tile(filter, shape1)

    shapeb = pshape.copy()
    shapeb[:] = 1
    shapeb[0] = 1
    shapeb[1] = pshape[1]

    # filter2 = np.ones(shapeb) * filter # is this right?
    # shape2 = pshape.copy()
    # shape2[1] = 1
    # filter2 = np.tile(filter2, shape2)
    # it seems this is what happens in matlab
    filter2 = filter1.transpose()
    newfilter = filter1 * filter2

    if dims == 3:
        filter3 = np.ones((1, 1, side2)) * filter[:, np.newaxis, np.newaxis]
        filter3 = np.tile(filter3, (side2, side2, 1))
        newfilter = newfilter * filter3

    # column vector
    factor = newfilter.flatten()
    return factor

def create_windows(pattern, mask, qbarrier, filter):
    dims, side2, fullsize, pshape, cshape = get_dims(pattern)
    filter = np.hanning(side2)
    filter = np.fft.fftshift(filter)
    pure_factor = create_filter(filter, pshape, side2, fullsize, dims)
    pure_factor = pure_factor * pure_factor

# 512x256
    mask_in_shape = mask.reshape(cshape)
    base_penalty = None

    if dims == 3:
        mask_3d = mask_in_shape[:side2, :side2, :side2]
        base_penalty = np.double((mask_3d > 0) + 1j * (mask_in_shape[:side2, side2:2 * side2, :side2] > 0))
    else:
        # mask_2d = mask_in_shape[:side2, :side2]
        # base_penalty = np.double((mask_2d > 0) + 1j * (mask_in_shape[:side2, side2:2 * side2] > 0))
        #mask_part1 = mask_in_shape[0:side2, 0:side2] > 0
        #mask_part2 = mask_in_shape[0:side2, 0:side2] > 0
        #base_penalty = np.where(mask_part1, 1.0, 0.0) + 1j * np.where(mask_part2, 1.0, 0.0)
        base_penalty = np.zeros((side2, side2))#, dtype=complex)

        mask1 = mask_in_shape[:side2, :side2] > 0
        #mask2 = mask_in_shape[:side2, side2:side2 + side2] > 0

        base_penalty[mask1] = 1.0
        # base_penalty[mask2] = 1j


    pure_reshaped = np.reshape(pure_factor, pshape)
    base_penalty = np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftn(np.fft.fftshift(base_penalty)) * pure_reshaped))

    # using complex:
    # from here on, the results tend to differ from the matlab code after a few decimal places
    # should be fine, though
    # one thing we MAY want to do is to force 1s from circa [101, 101] to [155, 155]
    #base_penalty[]

    base_penalty = np.concatenate((np.real(base_penalty), np.imag(base_penalty)))
    base_penalty = base_penalty.flatten(order='C')  # looks sort of ok with column major
    base_penalty[base_penalty < 1e-8] = 0
    base_penalty = 1.0 - base_penalty

    # filter out numerical inaccuracies again
    #base_penalty[base_penalty < 1e-8] = 0
    # filtering not really working hmm

    # more aggressive filtering
    # base_penalty[base_penalty < 1e-3] = 0
    base_penalty[base_penalty < 1e-3] = 0

    base_penalty = np.reshape(base_penalty, 2 * fullsize)

    filter = np.hanning(side2)
    filter = np.fft.fftshift(filter)

    factor = create_filter(filter, pshape, side2, fullsize, dims)
    factor = factor * factor
    factor[factor == 0] = np.finfo(float).eps

    # factor = factor.flatten() # ? redundant
    # NOTE: the matlab version required to add machine epsilon to 0-values in factor
    # this is not necessary in numpy (hopefully)

    # factor and basepenalty both look ok
    return factor, base_penalty



def create_proxop(diffx, penalty, ourlinp):
    diffxt = ourlinp(diffx, 2)
    level = -diffxt

    mask = np.logical_and(penalty == 0, diffxt >= 0)
    level[mask] = 0

    xlevel = ourlinp(level, 1)
    proxop = zero_tolerant_quad(penalty, -diffxt - level, diffxt)
    return proxop, diffxt, level, xlevel

def zero_tolerant_quad(p, p2, p3):
    p = p.flatten()
    op = lambda origx, t=None, grad=0: smooth_quad_diag_matrix(p, p2, p3, origx, t, grad)

    def smooth_quad_diag_matrix(q, q2, q3, origx, t, grad):
        pm = max(q)

        # q2 seems to be arbitrarily small, equalling zero basically
        x = origx - q2

        mask = q == 0

        q4 = q.copy()
        q4[q3 < 0] = pm

        q5 = q2.copy()
        q5[mask] = 0

        q6 = q.copy()

        q6[x < 0] = pm
        x[mask] = origx[mask]

        if t is None:
            g = q * x
            v = 0.5 * np.sum(g * x - q4 * q3 * q3)
        else:
            # prevx = x.copy()
            x = (1.0 / (t * q + 1)) * x
            g = x + q2
            v = 0.5 * np.sum(q * x * x - q4 * q3 * q3)

        if grad:
            return v, g
        else:
            return v

    return op

def diffpoisson(scale, y, basey, minval, absrefpoint, filter, qbarrier):
    # TODO: optimize
    y = y.flatten()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mask = np.zeros_like(y)
        mask = ~(y < 0 | np.isnan(y))
        rscale = 1.0 / scale
        filterrsq = 1.0 / (filter ** 2)

        baseyscaled = basey / rscale
        absrefpointscaled = absrefpoint * rscale

    f = lambda x, grad=0: diff_func(scale, rscale, mask, y, baseyscaled, minval, absrefpointscaled, filterrsq, qbarrier, x, grad)

    return f

def diff_func(scale, rscale, mask, y, base_y, minval, absrefpoint, filterrsq, qbarrier, x, grad=False, ret_vals=False):
    x = x * scale
    lim = qbarrier * filterrsq * (rscale * rscale)

    mask2 = mask.copy()
    mask2[:] = True
    # mask2 = np.ones(mask.shape, dtype=bool)

    lim[mask2] = np.maximum(lim[mask2], (2 * np.maximum(y[mask2], 1e-2) * lim[mask2]) ** 0.5)

    x_base = -base_y + minval - lim / 2
    upperlim = x_base + lim
    subupper = x < upperlim
    # TODO: check this
    x_upperlim = x.copy()
    x_upperlim[subupper] = upperlim[subupper]
    #x_upperlim = np.where(subupper, upperlim, x)#

    vals = np.zeros_like(x)

    # FIXME
    # should not be 0s?
    refpoint = absrefpoint - base_y
    refpoint_upperlim = refpoint.copy()
    refpoint_upperlim[refpoint < upperlim] = upperlim[refpoint < upperlim]
    absrefpointupperlim = refpoint_upperlim + base_y

    # Clamp the log_lambda part at xupperlim
    log_lambda = np.log((x_upperlim + base_y) / np.maximum(absrefpointupperlim, 0.5e-9))

    vals[mask] = - (y[mask] * np.log((x_upperlim[mask] + base_y[mask]) / np.maximum(absrefpointupperlim[mask], 0.5e-9))
            - 1 * (x[mask] - 1 * (absrefpoint[mask] - base_y[mask]))
            + (-(x_upperlim[mask] - x[mask]) * (y[mask] / np.maximum(upperlim[mask] + base_y[mask], 1e-15))
            + (refpoint_upperlim[mask] - refpoint[mask]) * (y[mask] / np.maximum(upperlim[mask] + base_y[mask], 1e-15))))

    lim2 = lim.copy()
    lim2[~mask] = lim2[~mask] * 0.5

    subs = x.flatten() < x_base + lim2.flatten()
    limfac = np.ones(mask.shape) / lim2

    vals[subs] = vals[subs] + (x[subs] ** 2) * limfac[subs]

    # compensate by quadratic form from absrefpoint position if any
    subs2 = refpoint.flatten() < (x_base + lim2).flatten()
    vals[subs2] = vals[subs2] - (refpoint[subs2] ** 2) * limfac[subs2]

    #subs3 = np.logical_xor(subs, subs2)
    #subs3 = subs ^ subs
    # actual minus
    subs3 = 1 * subs - 1 * subs2

    vals = vals + (1 * subs3 * (x_base + lim2) ** 2 + (-1 * subs * x + subs2 * (absrefpoint - base_y)) * 2 * (x_base + lim2)) * limfac
    v = np.sum(vals)

    if grad:
        g = y[mask] / np.maximum(x_upperlim[mask] + base_y[mask], 1e-15) - 1
        old_x = x.copy()
        x[:] = 0
        x[mask] = -g
        if np.any(subs):
            x[subs] = x[subs] + 2 * (old_x[subs] - x_base[subs] - lim2[subs]) ** 1 * limfac[subs]
        x = x * rscale

        if ret_vals:
            return v, x, vals
        else:
            return v, x
    else:
        return v
