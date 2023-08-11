import numpy as np

def get_dims(pattern):
    side2 = pattern.shape[0]
    fullsize = pattern.size

    if fullsize == side2 ** 3:
        dims = 3
        pshape = (side2, side2, side2)
        cshape = (side2, side2 * 2, side2)
    elif fullsize == side2 ** 2:
        dims = 2
        pshape = (side2, side2)
        cshape = (side2, side2 * 2)
    else:
        raise ValueError('Unknown dimensionality')

    return dims, side2, fullsize, pshape, cshape