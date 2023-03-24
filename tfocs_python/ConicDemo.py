import ConicSolver
import numpy as np
import ConicSolver as CS

# Test solver on simple unconstrained quadratic function
def TestQuadratic():
	N = 100
	c = np.random.randn(N, 1)
	print(c.shape)

	D0 = np.random.randn(N, N)

	D = D0 * np.transpose(D0) + 0.5 * np.identity(N)
	x_star = - np.divide(D, c)
	x0 = np.zeros(N)

	# almost certainly not correct dude what
	# f = lambda x: c * x + x * D * x/2  # hopefully correct
	f = lambda x: np.transpose(c) * x + np.transpose(x) * D * x/2  # hopefully correct
	def fun(x):
		print(x)
		print(x.shape)
		print(c.shape)  # shape (1, 100)
		print(D.shape)
		# print(np.transpose(c) * x)
		return f(x)

	def grad_f(x):
		c + D * x

	def smooth_func(x, grad):
		wrapper_objective(fun, grad_f, x, grad)

	# problem: x0 is only 0s
	# affine and projector are both null
	solver = CS.ConicSolver(smooth_func, None, None, x0)

	solver.restart = 100

	out = solver.solve()

	print(out)


# unsure
def wrapper_objective(f, g, x, grad=0):  # we just set it to NOT return gradient by default
	if grad:
		print("x value: " + str(x))
		return f(x), g(x)
	else:
		return f(x), None

def smooth_quad(P = np.identity(2), q = 0, r = 0, use_eig = 0):
	if np.isnumeric(P):
		if np.isvector(P):
			if any(P) < 0:  # in matlab this is checked twice ??????
				raise Exception("P must be convex or concave but not both")



TestQuadratic()