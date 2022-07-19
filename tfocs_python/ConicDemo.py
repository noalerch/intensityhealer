import ConicSolver
import numpy as np
import ConicSolver as CS

def TestQuadratic():
	N = 100
	c = np.random.randn(N, 1)

	D0 = np.random.randn(N, N)
	D = D0 * np.transpose(D0) + 0.5 * np.identity(N)
	x_star = - np.divide(D, c)
	x0 = np.zeros(N)

	f = lambda x: c * x + x * D * x/2  # hopefully correct
	grad_f = lambda x: c + D * x
	smooth_func = lambda x, grad: wrapper_objective(f, grad_f, x, grad)

	solver = CS.ConicSolver(smooth_func, None, None, x0)

	solver.restart = 100

	out = solver.solve()

	print(out)


# unsure
def wrapper_objective(f, g, x, grad=0):  # we just set it to return gradient by default
	if grad:
		return f(x), g(x)
	else:
		return f(x), None

def smooth_quad(P = np.identity(2), q = 0, r = 0, use_eig = 0):
	if np.isnumeric(P):
		if np.isvector(P):
			if any(P) < 0:  # in matlab this is checked twice ??????
				raise Exception("P must be convex or concave but not both")



TestQuadratic()