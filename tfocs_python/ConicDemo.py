import ConicSolver as CS
import numpy as np

def TestQuadratic():
	N = 100
	c = np.random.randn(N, 1)

	D0 = np.random.randn(N, N)
	D = D0 * np.transpose(D0) + 0.5 * np.identity(N)
	x_star = - np.divide(D, c)
	x0 = np.zeros(N)


def smooth_quad(P = np.identity(2), q = 0, r = 0, use_eig = 0):


