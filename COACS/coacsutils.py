import numpy as np

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
        # does not work
        # base_penalty = np.double((mask_2d > 0) + 1j * (mask_in_shape[:side2, side2:2 * side2] > 0))
        mask_part1 = mask_in_shape[0:side2, 0:side2] > 0
        mask_part2 = mask_in_shape[0:side2, 0:side2] > 0
        base_penalty = np.where(mask_part1, 1.0, 0.0) + 1j * np.where(mask_part2, 1.0, 0.0)


    pure_reshaped = np.reshape(pure_factor, pshape)
    base_penalty = np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftn(np.fft.fftshift(base_penalty)) * pure_reshaped))

    # is this necessary?
    # filter out numerical inaccuracies
    # NOTE: these results are different from matlab
    # due to the fact that matlab uses doubles
    # thought it seems that it was intended to use imaginary numbers
    base_penalty = np.concatenate((np.real(base_penalty), np.imag(base_penalty)))
    base_penalty = base_penalty.flatten()
    base_penalty[base_penalty < 1e-8] = 0
    base_penalty = 1.0 - base_penalty

    # filter out numerical inaccuracies again
    base_penalty[base_penalty < 1e-8] = 0

    base_penalty = np.reshape(base_penalty, 2 * fullsize)
    # NOTE: base_penalty just 1s at first run
    # does this change?

    filter = np.hanning(side2)
    filter = np.fft.fftshift(filter)

    factor = create_filter(filter, pshape, side2, fullsize, dims)
    factor = factor * factor + 0 * 1e-8

    # factor = factor.flatten() # ? redundant

    return factor, base_penalty



def diffpoisson(scale, y, basey, minval, absrefpoint, filter, qbarrier):
    mask = ~(y < 0) & ~np.isnan(y)
    rscale = 1.0 / scale
    filterrsq = 1.0 / (filter ** 2)
    baseyscaled = basey * rscale
    absrefpointscaled = absrefpoint * rscale

    f = lambda *args: diff_func(scale, rscale, mask, y, baseyscaled, minval, absrefpointscaled, filterrsq, qbarrier,
                                *args)
    return f

def diff_func(scale, rscale, mask, y, baseyscaled, minval, absrefpointscaled, filterrsq, qbarrier, x):
    x = x * scale
    lim = qbarrier * filterrsq * (rscale * rscale)

    mask2 = mask.copy()


