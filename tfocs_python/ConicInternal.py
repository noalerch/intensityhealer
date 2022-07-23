# Internal functions for ConicSolver library
# Author: Noa Lerch, nzlrch@protonmail.com

def projection_Rn(x, t=None, grad=0):
    if grad == 0:
        return 0
    else:
        if t is None:
            return 0, 0 * x
        else:
            return 0, x  # g := x

# def proximity_stack()