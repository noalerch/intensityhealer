import ConicSolver
import numpy as np
import ConicSolver as CS

# Test solver on simple UNCONSTRAINED quadratic function
# ! NOTE: COACS seems to only use constrained problems
# This test can probably be ignored
def TestQuadraticUnconstrained():
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
		print("x: " + str(x))
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

#
def TestQuadraticConstrained():
	# minimize_x c'x + x' Dx / 2
	# subject to || x ||_1 <= 10
	N = 256
	c = np.random.randn(N, 1)
	D0 = np.random.randn(N, N)
	D = D0 * np.transpose(D0) + 0.5 * np.identity(N)
	x0 = np.zeros(N)  # ? (N, 1)?

	# several ways to test this. we start with
	def f(x):
		#print("from f x = " + str(x))
		#print("from f c = " + str(c))
		np.transpose(c) * x + np.transpose(x) * D * x/2  # hopefully correct

	def grad_f(x):
		c + D * x

	def smooth(x, grad=0):
		print("fun = " + str(f))
		wrapper_objective(f, grad_f, x, grad)

	def linear(x):   # { 1 ; 1 } however that should be represented
		return x

	# proj_l1(10): l1 ball, radius 10 (e.g. norm(x,1) <= 10)
	def projector(x):
		return proj_l1(10, x)  # ?

	solver = CS.ConicSolver(smooth, linear, projector, x0)

	#### test parameters
	solver.tolerance = 1e-16
	solver.max_iterations = 3000
	solver.restart = 100


	out = solver.solve()

	print(out)

# Assume single argument (in fact, assume 10)
# we get @(varargin)proj_l1_q(q,varargin{:})
# this is the proj_l1_q with 2 arguments, i.e. q and x
def proj_l1(q, x):
	v = 0
	if np.linalg.norm(x, 1) > q:
		v = float('inf')

	return v

# unsure
def wrapper_objective(f, g, x, grad=0):  # we just set it to NOT return gradient by default
	if grad:
		print("f(x) value: " + str(f(x)))
		# FIXME: f does not exist??
		print("f " + str(f))
		return f(x), g(x)
	else:
		print("no gradient")
		return f(x), None # remove None?

def smooth_quad(P = np.identity(2), q = 0, r = 0, use_eig = 0):
	if np.isnumeric(P):
		if np.isvector(P):
			if any(P) < 0:  # in matlab this is checked twice ??????
				raise Exception("P must be convex or concave but not both")




TestQuadraticConstrained()