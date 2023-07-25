import unittest
import numpy as np
import ConicSolver as cs


class InitTest(unittest.TestCase):
    def setUp(self):
        self.N = 256
        self.c = np.random.randn(self.N, 1)
        self.D0 = np.random.randn(self.N, self.N)
        self.D = self.D0 * np.transpose(self.D0) + 0.5 * np.identity(self.N)
        self.x0 = np.zeros(self.N)  # ? (N, 1)?

    def test_init_smooth(self):
        solver = cs.ConicSolver(self.smooth, self.linear, self.projector, self.x0)
        assert self.x0.shape == (256,)

        A_y = solver.iv.A_y
        print(A_y)
        A_y_application = solver.apply_smooth(A_y, grad=0)  # non existent
        A_y_application_grad = solver.apply_smooth(A_y, grad=1)

        self.assertIsNotNone(A_y_application)  # add assertion here
        self.assertIsNotNone(A_y_application_grad)  # add assertion here

    def test_init_linear(self):
        solver = cs.ConicSolver(self.smooth, self.linear, self.projector, self.x0)
        # g_Ay = solver.iv.g_Ay
        # print(g_Ay)
        # g_Ay_application = solver.apply_linear(g_Ay, 2)
        self.assertIsNotNone(solver.apply_linear)
        # self.assertIsNotNone(g_Ay_application)  # add assertion here

    def test_init_projector(self):
        solver = cs.ConicSolver(self.smooth, self.linear, self.projector, self.x0)
        out = solver.solve()

    def test_run_solver_quad_unconstrained(self):
        solver = cs.ConicSolver(self.smooth, self.linear, self.projector, self.x0)
        out = solver.solve()

    # ---------------------------------------------------- #
    # ----- Functions for global use in solver tests ----- #
    def smooth(self, x, grad=0):
        def fun(y):
            ret = np.transpose(self.c) * y + np.transpose(y) * self.D * y / 2  # hopefully orrect
            return ret

        def grad_fun(y):
            return self.c + self.D * y

        def wrapper_objective(f, g, y, gradient=0):  # we just set it to NOT return gradient by default
            if gradient:
                return f(y), g(y)
            else:
                return f(y)

        def smooth_stack(functions, x_smooth, gradient=0):
            n = len(functions)
            #if n == 1:
            #    functions = functions[0]

            if n == 0:
                raise Exception("smooth zero not implemented yet!")
            elif callable(functions):
                return functions
            else: # smooth_stack_impl
                def smooth_stack_impl(smth, ecks, grd=0):
                    if grd:
                        f = [i for i in range(n)]
                        g = [0 for _ in smth] # prob wrong
                        for k in range(n):
                            f[k], g[k] = smth[k](ecks[k])
                        f = sum(f)
                        return f, g
                    else:
                        f = 0
                        for k in range(n):
                            f = f + smth[k](ecks[k])
                        return f
                return smooth_stack_impl(functions, x_smooth, gradient)  # (functions, x_smooth, gradient)

        # TODO: proper smooth stack usage

        # in tfocs, this stacks wrapper objective and smooth linear? which seems to result in dot?
        wrapper = lambda w: wrapper_objective(fun, grad_fun, w, grad)
        stacked = smooth_stack([wrapper], x, grad)
        return stacked



    # offset correct?
    def linear(self, x, mode):  # { 1 ; 1 } however that should be represented

        print("inside linear, highly suspicious")
        print(x)
        assert len(x) > 0

        def linop_stack_col(linear_col, N, dims, x_linop, mode_linop):
            y = 0
            if mode_linop == 0:
                print("mode 0")
                y = dims
            elif mode_linop == 1:  # tuple mode?
                print("mode 1")
                # y = (np.array(self.N), np.array(self.N))
                # y = (x_linop, x_linop)
                yl = [None, None]
                for j in range(2):
                    lf = linear_col[j]
                    if lf is not None:
                        yl[j] = lf(x_linop, 1)
                    else:
                        yl[j] = np.zeros_like(x_linop)
                y = (yl[0], yl[1])

                # lf = linop_identity
                # if lf is not None:
                #    y[j] = lf(x_linop, 1)
                # else:
                #    y[j] = np.zeros_like(x_linop)
                # raise Exception("linop stack mode unexpected!")
                # y = tfocs_tuple(y)
            elif mode_linop == 2:  # float mode?
                print("mode 2")
                # y = 0
                #  x_list = list(x_linop) # ???
                # for j in range(N): # n = 2
                #    lF = linearF[j]
                #    if lF is not None:
                #        print(x_linop)
                # this is the linear (identity) operator
                # called with mode 2, it will just give the identity
                # of x_list[j]
                #        y = y + lF(x_list[j], 2)
                # hack since we know what linearF is in our case
                y = x_linop[0] + x_linop[1]
                # assert (len(x_linop) == 2)
            return y

        # linearF = { @linop_identity, 1 }
        # m = 2
        # inp_dims = []
        # otp_dims = []

        # linearF = (linop_identity, 1)

        def linop_identity(x_linop, _=1):
            return x_linop

        def linop_dot_forward(sz, A, x_dot, mode_dot):
            if mode_dot == 0:
                return sz
            elif mode_dot == 1:
                # tfocs_dot in original
                # which is actually an optimized implementation
                # with support for complex numbers
                # assume np.dot behaves the same for our purposes
                # FIXME: it probably doesn't lol
                if np.isscalar(A) and not np.isscalar(x_dot):
                    if A == 0:
                        return 0
                    else:
                        return A * np.sum(x_dot)
                elif not np.isscalar(A) and not np.isscalar(x_dot):
                    return np.dot(A, x_dot)
                else: raise Exception("not implemented")
            elif mode_dot == 2:
                if not np.isreal(x):
                    raise Exception("Unexpected complex")
                return A * x_dot
            else:
                raise Exception("Invalid mode for linop")


        # ignore lambda for dot?
        sz = ([1, 1], [1, 1])
        a = 1
        linop_dot = lambda x_d, mode_d: linop_dot_forward(sz, a, x_d, mode_d)
        linearF = (linop_identity, linop_dot)

        return linop_stack_col(linearF, 2, ([], []), x, mode)

    # supposedly projector should take step parameter
    def projector(self, x, step=None):
        print("SHAPe ################################## of x")
        print(x.shape)
        return self.proj_l1(10, x, step)  # ?

    # proj_l1(10): l1 ball, radius 10 (e.g. norm(x,1) <= 10)
    def proj_l1(self, q, x, step):
        # changed norm params
        x_old = np.copy(x)
        v = 0
        if step is None:
            if np.linalg.norm(x) > q:
                v = float('inf')
            return v
        else:
            x_return = np.copy(x)
            print("x from inside proj_l1 ", x.shape)
            s = np.sort(np.abs(x[np.where(x != 0)]))
            cs = np.cumsum(s)
            # ndx = find(cs - (1:numel(s))' .* [ s(2:end) ; 0 ] >= q+2*eps(q), 1 )
            s_shifted = np.concatenate((s[1:], [0]))
            diff = cs - np.arange(1, len(s) + 1) * s_shifted
            ndx = np.where(diff >= q + 2 * np.finfo(float).eps)[0]
            if len(ndx) > 0:
                ndx = ndx[0]
                thresh = (cs[ndx] - q) / (ndx + 1)
                x_return = x * (1 - thresh / np.maximum(np.abs(x), thresh))
            assert (x.all() == x_old.all())
            print("SHAPe ##################################")
            print(x_return.shape)
            return v, x_return

    # return value?
    def smooth_quad(self, P=np.identity(2), q=0, r=0, use_eig=0):
        if np.isnumeric(P):
            if np.isvector(P):
                if any(P) < 0:  # in matlab this is checked twice ??????
                    raise Exception("P must be convex or concave but not both")


if __name__ == '__main__':
    unittest.main()
