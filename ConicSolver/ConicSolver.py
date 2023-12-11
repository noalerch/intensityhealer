import math
import numpy as np


# TODO: affine_func, projector_func are optional
class ConicSolver:
    def __init__(self): #, smooth_func, affine_func, projector_func, x0) -> None:
        """ConicSolver constructor
        """

        # instance attributes taken from tfocs_initialize.m
        self.apply_linear = None
        self.apply_smooth = None
        self.apply_projector = None
        self.max_iterations = float('inf')
        self.max_counts = float('inf')
        self.count_ops = False
        self.count = np.array([0, 0, 0, 0, 0])
        self.save_history = True
        self.adjoint = False
        self.saddle = False
        self.tolerance = 1e-8
        self.error_function = None
        self.stop_function = None
        self.print_every = 100
        self.max_min = 1
        self.beta = 0.5
        self.alpha = 0.9
        self.L_0 = 1
        self.L_exact = float('inf')
        self.L_local = 0  # changes with backtrack
        self.mu = 0
        self.fid = 1
        self.stop_criterion = 1
        self.alg = 'AT'
        self.restart = float('inf')
        self.print_stop_criteria = False
        self.counter_reset = -50
        self.cg_restart = float('inf')
        self.cg_type = 'pr'
        self.stop_criteria_always_use_x = False
        self.data_collection_always_use_x = False
        self.output_always_use_x = False
        self.auto_restart = 'gra'  # function or gradient
        self.print_restart = True
        self.debug = False

        self.iv = IterationVariables()

        # iterations start at 0
        self.n_iter = 0

        self.test = []
        self.output_dims = 256
        self.L = self.L_0
        self.theta = float('inf')
        self.f_v_old = float('inf')
        self.f_v = None  # objective value
        self.xy_sq = 0
        # the way this works in TFOCS is that affineF
        # is a cell array of an arbitrary amount of
        # linear functions each paired with an offset

        # assume false
        if self.adjoint:
            print("adjoint not implemented!")

        self.restart_iter = 0
        self.warning_lipschitz = False  # 0 in matlab
        self.backtrack_simple = True
        self.backtrack_tol = 1e-10
        self.backtrack_steps = 0

        self.just_restarted = False

        self.output = SolverOutput()

        self.cs = None

    def solve(self, smooth_func, affine_func, projector_func, x0, affine_offset):
        """

        assumes Auslender-Teboulle algorithm for now
        """

        # we set the functions here to allow for greater flexibility
        # in choosing options prior to solving
        self.set_smooth(smooth_func, affine_offset)
        self.set_linear(affine_func)
        self.set_projector(projector_func)

        # iv = IterationVariables()
        # self.iv = IterationVariables()
        self.iv.output_dims = 256  # default value = 256
        self.iv.init_x(x0)
        self.iv.z = x0  # suspicious
        self.iv.y = x0  # suspicious

        self.iv.x = x0

        self.L = self.L_0

        if np.isinf(self.iv.C_x):
            self.iv.C_x = self.apply_projector(self.iv.x)
            if np.isinf(self.iv.C_x):  # still inf? get gradient
                self.iv.C_x, self.iv.x = self.apply_projector(self.iv.x, 1)

        size_ambig = True
        if size_ambig:
            # pretty good but slightly wrong
            self.iv.A_x = self.apply_linear(self.iv.x, 1)  # correct so far
            # is this where it explodes?
        else:
            self.iv.A_x = np.zeros((self.output_dims, self.output_dims))

        # circa line 521 in tfocs_init
        self.iv.f_x, self.iv.g_Ax = self.apply_smooth(self.iv.A_x, grad=1)
        if np.isinf(self.iv.f_x):
            raise Exception("The initial point lies outside of the domain of the smooth function.")

        # initialize values for y and z
        self.iv.init_iterate_values()

        # call AT function to optimize
        self.print_header("Auslender & Teboulle's single-projection method")
        self.auslender_teboulle()

        return self.iv.x, self.output

    def print_header(self, alg_name=None):
        if self.print_every:
            if alg_name is None:
                algorithm = self.alg  # Replace with your algorithm name
            else:
                algorithm = alg_name
            print(algorithm)
            print('Iter      Objective   |dx|/|x|    step', end='')
            if self.count_ops:
                print('       F     G     A     N     P', end='')
            if self.error_function:
                nBlanks = max(0, len(self.error_function) * 9 - 9)
                print(f'      errors{" " * nBlanks}', end='')
            if self.print_stop_criteria:
                print('    stopping criteria', end='')
            print()

            print('----+----------------------------------', end='')
            if self.count_ops:
                print('+-------------------------------', end='')
            if self.error_function:
                print(f'+{"-" * (1 + len(self.error_function) * 9)}', end='')
            if self.print_stop_criteria:
                print(f'+{"-" * 19}', end='')
            print()

    def auslender_teboulle(self):  # we pass iv as an argument to avoid using self.iv
        """Auslender & Teboulle's method
        args:
            smooth_func: function for smooth
            affine_func: function for smooth

        """

        # following taken from tfocs_initialize.m
        # theta = float('inf')
        # self.f_v_old = float('inf')
        counter_Ay = 0
        counter_Ax = 0

        while True:
            x_old = self.iv.x
            z_old = self.iv.z
            A_x_old = self.iv.A_x
            A_z_old = self.iv.A_z

            # backtracking loop
            L_old = self.L
            self.L = self.L * self.alpha
            theta_old = self.theta

            while True:
                # acceleration
                self.theta = self.advance_theta(theta_old)  # use L args?

                # next iteration
                if self.theta < 1:
                    self.iv.y = (1 - self.theta) * x_old + self.theta * z_old

                    if counter_Ay >= self.counter_reset:
                        # A_y = self.apply_linear(y, 1)
                        # iv.A_y = self.apply_linear(iv.y, 2) # why is this 2??
                        self.iv.A_y = self.apply_linear(self.iv.y, 1)

                        counter_Ay = 0  # Reset counter

                    else:
                        counter_Ay += 1
                        self.iv.A_y = (1 - self.theta) * A_x_old + self.theta * A_z_old

                    # UPDATE: moved into here
                    self.iv.f_y = float('inf')
                    self.iv.g_Ay = np.array([])
                    self.iv.g_y = np.array([])

                if self.iv.g_y.size == 0:
                    # initialize g_Ay
                    if self.iv.g_Ay.size == 0:
                        self.iv.f_y, self.iv.g_Ay = self.apply_smooth(self.iv.A_y, grad=1)

                    self.iv.g_y = self.apply_linear(self.iv.g_Ay, 2)

                step: float = 1 / (self.theta * self.L)


                # C_z correct, z slightly off
                self.iv.C_z, self.iv.z = self.apply_projector(z_old - step * self.iv.g_y, t=step, grad=1)

                # A_z looks correct
                self.iv.A_z = self.apply_linear(self.iv.z, 1)

                # new iteration
                if self.theta == 1:
                    self.iv.x = self.iv.z
                    self.iv.A_x = self.iv.A_z
                    self.iv.C_x = self.iv.C_z

                else:
                    self.iv.x = (1 - self.theta) * x_old + self.theta * self.iv.z

                    if counter_Ax >= self.counter_reset:
                        counter_Ax = 0
                        self.iv.A_x = self.apply_linear(self.iv.x, 1)
                    else:
                        counter_Ax += 1
                        self.iv.A_x = (1 - self.theta) * A_x_old + self.theta * self.iv.A_z

                    self.iv.C_x = float('inf')

                self.iv.f_x = float('inf')

                self.iv.g_Ax = np.array([])
                self.iv.g_x = np.array([])

                break_val, counter_Ax = self.backtrack(counter_Ax)
                if break_val:
                    break

            # TODO: proper implementation of xy if we want to handle
            #       stopCrit 2
            break_val, v_is_x, v_is_y, f_vy, status = self.iterate(x_old, A_x_old)
            if break_val:
                break

        self.cleanup(v_is_x, v_is_y, f_vy, status)

    def cleanup(self, v_is_x, v_is_y, f_vy, status):
        # TODO: cur_dual (probably not needed for COACS)
        n_iter = self.n_iter

        if v_is_y and not self.output_always_use_x \
                and not self.data_collection_always_use_x:
            f_vy = self.f_v

        if not v_is_x:
            if self.saddle:
                if self.iv.g_Ax is None:
                    self.iv.f_x, self.iv.g_Ax = self.apply_smooth(self.iv.A_x, grad=1)


            elif np.isinf(self.iv.C_x):
                self.iv.C_x = self.apply_projector(self.iv.x)

            self.f_v = self.max_min * (self.iv.f_x + self.iv.C_x)
            cur_pri = self.iv.x

        # take whichever of x or y is better
        x_or_y_string = 'x'
        if v_is_y and not self.output_always_use_x and f_vy < self.f_v:
            self.f_v = f_vy

            self.iv.x = self.iv.y
            x_or_y_string = 'y'

        # ignoring saddle points

        if self.fid and self.print_every:
            print("Finished: %s\n" % status)  # , file=self.fid)

        self.output.n_iter = self.n_iter
        self.output.status = status
        self.output.x_or_y = x_or_y_string


        if self.save_history:
            self.output.f[n_iter] = self.f_v

            # this just clearing an array?
            # self.output.f[n_iter:end] = [] # TODO fix this

            # i think what we want is to delete (np.delete()) the elements
            # of the array after n_iter
            self.output.f = self.output.f[:n_iter]  # only the first n_iter elements
            self.output.norm_grad = self.output.norm_grad[:n_iter]
            self.output.theta = self.output.theta[:n_iter]  # assuming numpy works this way

            if self.count_ops:
                self.output.counts = self.output.counts[:n_iter]

                # assume empty error function

                # TODO: descriptions

                if self.count_ops:
                    self.count = np.array([0, 0, 0, 0, 0])

    # based on tfocs_iterate.m script
    def iterate(self, x_old, A_x_old):
        status = ""

        v_is_x = False
        v_is_y = False

        # this kind of defeats the purpose of data_collection_always_use_x
        # but this is a simple solution
        # NOTE: this MIGHT lead to bugs!
        f_vy = self.f_v

        # test for positive stopping criteria
        self.n_iter += 1
        norm_x = np.linalg.norm(self.iv.x)
        norm_dx = np.linalg.norm(self.iv.x - x_old)

        # xy_sq = 0  # placeholder

        # legacy stopping criteria
        # not necessary for jackdaw-based COACS
        if self.stop_criterion == 2 and self.beta >= 1:
            xy = self.iv.x - self.iv.y

            self.xy_sq = self.square_norm(xy)

        current_dual = None

        limit_reached = False  # bool nicer than the string search in tfocs

        # could perhaps use match-case which was introduced in Python 3.10
        # avoiding this due to compatibility issues
        if np.isnan(self.iv.f_y):
            status = "NaN found -- aborting"
        elif self.stop_criterion == 1 and norm_dx == 0:
            if self.n_iter > 1:
                status = "Step size tolerance reached (||dx||=0)"
        elif self.stop_criterion == 1 and norm_dx < self.tolerance * max(norm_x, 1):
            status = "Step size tolerance reached"
        elif self.stop_criterion == 2 and self.L * math.sqrt(xy_sq) < self.tolerance * max(norm_x, 1):
            status = "Step size tolerance reached"
        elif self.n_iter == self.max_iterations:
            status = "Iteration limit reached"
            limit_reached = True
        elif self.count_ops and np.max(self.count) >= self.max_counts:
            # TODO: make wrapper functions for counting operations on function application
            status = "Function/operator count limit reached"
            limit_reached = True
        elif self.backtrack_steps > 0 and self.xy_sq == 0:
            status = f"Unexpectedly small step size after {self.backtrack_steps} backtrack steps"

        # for stop_crit 3, need new and old dual points
        # TODO most of this. Left for now because not needed for COACS
        if self.stop_criterion == 3 or self.stop_criterion == 4:
            if not self.saddle:
                raise "stop criterion {3, 4} requires a saddle point problem"

        # Use function value for y instead of x if cheaper
        # This part determines computational cost before continuing

        # Honestly unsure if any of these conditions will ever be true in COACS
        # particularly in the second clause
        if (status == "" or limit_reached) and (self.stop_function is not None
                                                or self.restart < 0 or self.stop_criterion in [3, 4]):

            need_dual = self.saddle and (self.stop_function is None or
                                         self.stop_criterion in [3, 4])

            # unsure of these tfocs_iterate.m lines 60-1
            # TODO: we may run into errors and unexpected behavior
            #       between 0-1 ints, bools, and arrays of bools
            comp_x = np.array([np.isinf(self.iv.f_x), need_dual * bool(self.iv.g_Ax.size), np.isinf(self.iv.C_x)])
            comp_y = np.array([np.isinf(self.iv.f_y), need_dual * bool(self.iv.g_Ay.size), np.isinf(self.iv.C_y)])

            if np.sum(comp_x) <= np.sum(comp_y) or self.stop_criteria_always_use_x:

                if comp_x[1]:
                    self.iv.f_x, self.iv.g_Ax = self.apply_smooth(self.iv.A_x, grad=1)
                elif comp_x[0]:
                    self.iv.f_x = self.apply_smooth(self.iv.A_x)
                if comp_x[2]:
                    self.iv.C_x = self.apply_projector(self.iv.x)

                current_priority = self.iv.x
                if self.saddle:
                    current_dual = self.iv.g_Ax
                self.f_v = self.max_min * (self.iv.f_x + self.iv.C_x)
                v_is_x = True

            else:

                if comp_y[1]:
                    self.iv.f_y, self.iv.g_Ay = self.apply_smooth(self.iv.A_y, grad=1)
                elif comp_y[0]:
                    self.iv.f_y = self.apply_smooth(self.iv.A_y)

                current_priority = self.iv.y
                if self.saddle:
                    current_dual = self.iv.g_Ay

                self.f_v = self.max_min * (self.iv.f_y + self.iv.C_y)
                v_is_y = True

                if self.data_collection_always_use_x:
                    f_vy = self.f_v
                    if self.saddle:
                        dual_y = current_dual

            # TODO: llnes 84-96 in tfocs_iterate.m
            #       likely unnecessary for COACS
            # raise RuntimeWarning("Unexpected! Please implement lines 84 from tfocs_iterate.m")

        # TODO: finish this part
        #       i cannot remember why this TODO exists. remove?
        #    comp_x = [np.isinf(f_x), need_dual]

        # TODO: apply stop_criterion 3 if it has been requested
        #       not yet implemented since COACS uses default stop_crit

        # Data collection
        # fid
        will_print = self.fid and self.print_every and (status != ""
                                                        or self.n_iter % self.print_every != 0
                                                        or (self.print_restart and self.just_restarted))

        if self.save_history or will_print:

            if (self.data_collection_always_use_x and not v_is_x) or (not v_is_x and not v_is_y):

                f_x_save = self.iv.f_x
                g_Ax_save = self.iv.g_Ax

                # both of these should be false for COACS
                if self.error_function is not None and self.saddle:

                    if self.iv.g_Ax is not None:
                        self.iv.f_x, self.iv.g_Ax = self.apply_smooth(self.iv.A_x, grad=1)

                    current_dual = self.iv.g_Ax

                if np.isinf(self.iv.f_x):
                    self.iv.f_x = self.apply_smooth(self.iv.A_x)

                if np.isinf(self.iv.C_x):
                    self.iv.C_x = self.apply_projector(self.iv.x)

                self.f_v = self.max_min * (self.iv.f_x + self.iv.C_x)
                v_is_x = True

                # Undo calculations
                self.iv.f_x = f_x_save
                self.iv.g_Ax = g_Ax_save

            # TODO: we ignore error and stop functions

        # iterate line 226
        if status == "" and self.beta < 1 and self.backtrack_simple \
                and self.L_local > self.L_exact:
            self.warning_lipschitz = True

        if will_print:
            if self.warning_lipschitz:
                self.warning_lipschitz = False
                bchar = 'L'

            elif self.backtrack_simple:
                bchar = ' '

            else:
                bchar = '*'

            # TODO: format may be incorrect
            to_print = "('%-4d| %+12.5e  %8.2e  %8.2e%c)" % (self.n_iter, self.f_v,
                                                                 norm_dx / max(norm_x, 1), 1 / self.L, bchar)

            print(to_print, end="")

            if self.count_ops:
                print("|", end="")

                # TODO: tfocs_count___ is array??
                # print(f"%5d", self.count)  # , file=self.fid)
                print(self.count, end="")  # , file=self.fid)

            if self.error_function is not None:
                if self.count_ops:
                    print(' ', end="")  # , file=self.fid)

                print('|', end="")  # , file=self.fid)
                # TODO: no errs since error function is null by default
                #       thus, ignore for now

            # display number used to determine stopping
            # in COACS this should always be 1
            if self.print_stop_criteria:
                if self.stop_criterion == 1:
                    if norm_dx is not None and norm_x is not None:
                        stop_resid = norm_dx / max(norm_x, 1)

                    else:
                        stop_resid = float('inf')

                else:
                    raise Exception(f"stop criterion {self.stop_criterion} not yet implemented")

                if self.error_function is not None or self.count_ops:
                    print(' ')  # , file=self.fid)

                print('|')  # , file=self.fid)

                # assumes stop_resid exists (i. e. stop_criterion == 1)
                print(" %8.2e", stop_resid)  # , file=self.fid) # hopefully correct syntax

            if self.print_restart and self.just_restarted:
                print(' | restarted', end="")  # , file=self.fid)

            print('\n')  # , file=self.fid)

        # extending arrays if needed
        if self.save_history:
            f_size = self.output.f.size
            if f_size < self.n_iter and status == "":
                csize = int(min(self.max_iterations,
                            f_size + 1000))  # this is +1 compated to TFOCS due to matlab indexing. Does this matter?

                # set values from end to czise to 0

                # removed + 1 because of 0-indexing
                # csize = 601
                self.output.f = np.pad(self.output.f, (0, csize))  # TODO: verify
                self.output.theta = np.pad(self.output.theta, (0, csize))  # TODO: verify
                self.output.step_size = np.pad(self.output.step_size, (0, csize))  # TODO: verify
                self.output.norm_grad = np.pad(self.output.norm_grad, (0, csize))  # TODO: verify

                if self.count_ops:
                    # uses : instad of 1 in matlab code. Please check!
                    self.output.norm_grad = np.pad(self.output.norm_grad, (0, csize))  # TODO: verify

                # TODO: check indexing

        if status == "":
            do_break = False
        else:
            do_break = True

        # tentative attempt att iterate lines 330--

        self.backtrack_steps = 0
        self.just_restarted = False
        do_auto_restart = False
        if self.restart < 0:
            if self.auto_restart == 'gra':
                do_auto_restart = np.dot(self.iv.g_Ay, self.iv.A_x - A_x_old)

            elif self.auto_restart == 'fun':
                do_auto_restart = self.max_min * self.f_v > self.max_min * self.f_v_old

        if self.n_iter - self.restart_iter == abs(round(self.restart)) \
                or do_auto_restart:
            self.restart_iter = self.n_iter
            self.backtrack_simple = True
            self.theta = float('inf')

            self.iv.reset_yz()

            self.f_v_old = self.max_min * float('inf')

            self.just_restarted = True

        # do_auto_restart

        return do_break, v_is_x, v_is_y, f_vy, status

    # based on Nettelblad's changed backtracking logic for TFOCS
    # handles numerical errors better
    def backtrack(self, counter_Ax):

        # instead of setting a do_break variable (which
        # is always done in the context of a break/return
        # in tfocs for matlab version >= R2015b
        # we simply return True and account for this in the
        # call to this function
        if self.beta >= 1:
            return True, counter_Ax


        xy = self.iv.x - self.iv.y

        self.xy_sq = self.square_norm(xy)

        if self.xy_sq == 0:
            self.L_local = float('inf')
            return True, counter_Ax

        # to handle numerical issues from the ratio being smaller than machine epsilon
        # force reset
        if self.xy_sq / (self.square_norm(self.iv.x)) < np.finfo(float).eps:
            counter_Ax = float('inf')
            return True, counter_Ax

        if self.iv.g_Ax.size == 0 or np.isinf(self.iv.f_x):
            # for projector and full solver test
            # A_x :: 2-tuple of 256-vectors
            # A_x in TFOCS:
            # {(100,1) double, [-10]}
            # error smooth is broken
            # f_x supposed to be scalar
            # A_x not supposed to be 0
            # FIXME
            self.iv.f_x, self.iv.g_Ax = self.apply_smooth(self.iv.A_x, grad=1)

        # in tfocs_backtrack it simply overwrites backtrack_simple
        # before changing again in the next lines
        within_tolerance = abs(self.iv.f_y - self.iv.f_x) >= \
                           self.backtrack_tol * max(max(abs(self.iv.f_x), abs(self.iv.f_y)), 1)

        # .^ is in matlab elementwise power, we represent as **
        self.backtrack_simple = within_tolerance and (abs(self.xy_sq) >= self.backtrack_tol ** 2)

        # assuming np.dot is equivalent to tfocs_dot
        L_local_origin = 2 * np.dot(self.iv.A_x - self.iv.A_y, self.iv.g_Ax - self.iv.g_Ay) / self.xy_sq

        self.L_local = max(self.L, L_local_origin)

        q_x = np.dot(xy, self.iv.g_y + 0.5 * self.L * xy)

        L_local_2 = self.L + 2 * max((self.iv.f_x - self.iv.f_y) - q_x + max(
            [np.finfo(self.iv.f_x).eps, np.finfo(self.iv.f_y).eps, np.finfo(q_x).eps,
             np.finfo(self.iv.f_x - self.iv.f_y).eps]), 0) / self.xy_sq

        if self.backtrack_simple:
            self.L_local = min(self.L_local, L_local_2)

        # NOTE: that normlimit in nettelblads backtrack is only called from
        #       code which is already commented out
        # norm_limit = np.array([abs(xy_sq) / (self.backtrack_tol * max(max(abs(np.dot(x, x)), abs(np.dot(y, y))), 1)))])

        self.backtrack_steps += 1

        if self.iv.f_x - self.iv.f_y > 0:
            self.L_local = max(self.L, self.L_local)

        if self.L_local <= self.L or self.L_local >= self.L_exact:
            return True, counter_Ax  # analogous to break in matlab script?

        # if np.isinf(self.L_local):
        #    pass
        elif self.L_local == float('inf'):
            self.L_local = self.L

        # FIXME: L is wrong
        self.L = min(self.L_exact, self.L / self.beta)

        return False, counter_Ax

    # TODO? ignore for now
    # def apply_linear(self, mode):
    #    pass
    # this can't be right lol
    # return self.solver_apply(3, self.linear_function, x, mode)

    # calculating gradient may be more expensive than just getting value
    # at x, so TODO: make gradient optional.
    # perhaps two different functions?

    # The linear/smooth/projector functions work such that
    # the optional argument grad determines whether or not
    # the gradient should be provided along with the main
    # value. if grad=1, then return 2-tuple of main value
    # and gradient

    # a note of the implementation linearF of a linear operator A:
    # y = linearF(x, mode)
    # mode=0: return size of linear operator, ignore x
    # mode=1: apply forward operation y = A(x)
    # mode=2: apply adjoint operation y = A*(x) (that is A-star, A*)
    def set_linear(self, linear_func):
        if self.count_ops:
            #self.apply_linear = lambda x, mode, grad=0: self.solver_apply(2, linear_func, x, mode, grad)
            def apply_linear(x, mode, grad=0):
                self.count[2] += 1

                if grad == 0:
                    return linear_func(x, mode)
                else:
                    return linear_func(x, mode, grad)

            self.apply_linear = apply_linear

        else:
            # self.apply_linear = lambda *args: linear_func(*args)
            self.apply_linear = linear_func


    def set_smooth(self, smooth_func, offset=0):
        if self.count_ops:
            # code ugly and unreadable
            #self.apply_smooth = lambda x, grad=0: self.solver_apply([i for i in range(0, 1 + grad)],
            #                                                        smooth_func, [x + offset, grad])
            def apply_smooth(x, grad=0):
                self.count[0] += 1
                return smooth_func(x + offset, grad)

            self.apply_smooth = apply_smooth

        else:
            #self.apply_smooth = lambda x, *args: smooth_func(x + offset, *args)
            self.apply_smooth = lambda x, grad=0: smooth_func(x + offset, grad)

    def set_projector(self, projector_func):
        if projector_func is None:
            projector = self.projection_Rn

        else:
            projector = projector_func  # proximity_stack(projector)
        self.apply_projector = projector  # lambda args, grad = 0: self.solver_apply([i for i in range(3, 4 + grad)], projector, args)


    def projection_Rn(self, x, t=None, grad=0):
        if grad == 0:
            return 0
        else:
            if t is None:
                return 0, 0 * x
            else:
                return 0, x  # g := x

    # deprecated
    def solver_apply(self, ndxs, func, val, mode, g=0):
        self.count[ndxs] += 1
        return func(val, mode, g)

    # assumes mu > 0 & & ~isinf(Lexact) && Lexact > mu,
    # see tfocs_initialize.m (line 532-) and healernoninv.m
    def advance_theta(self, theta_old: float):
        ratio = math.sqrt(self.mu / self.L_exact)
        theta_scale = (1 - ratio) / (1 + ratio)
        return min(1.0, theta_old * theta_scale)

    def square_norm(self, arr):
        return math.sqrt(np.dot(arr, arr))


# TODO: finish output stuff
class SolverOutput:
    # default output values at start
    def __init__(self):
        self.alg = "AT"
        self.f = np.array([])
        self.theta = np.array([])
        self.step_size = np.array([])
        self.norm_grad = np.array([])
        self.counts = np.array([])
        self.x_or_y = None
        self.dual = None

    def display(self):
        print("Display not implemented!")


class IterationVariables:
    def __init__(self):
        # construct common initial values

        # x values
        self.x = np.array([])  # this never seems to be updated?
        self.A_x = np.array([])  # Non-ambiguous output dimensions: A_x = zeros
        # n_smooth = numel(smoothF)
        # assumtion 23-3-29: assume smooth function is singular, i.e. numel(smoothF) = 1
        # output dimensions are a cell array

        self.f_x = float('inf')
        self.C_x : float = float('inf')
        self.g_x = np.array([])
        self.g_Ax = np.array([])

        # attempt 1
        # todo?

        # attempt 2

        # y values
        # self.output_dims = 0
        # default value
        self.output_dims = 256

        ### size_ambig:
        # sz :: double => if sz empty
        # sz :: cell => false if valid function handle
        # sz :: cell => true if empty
        # for each element in sz, if at least one contained element ambiguous => true
        # sz :: any other type => true

        # if isempty( A_x ),
        #    if identity_linop || zero_x0 && square_linop, # identity_linop??
        #        A_x = x;
        #    elseif ~zero_x0 || size_ambig( otp_dims ),  % Jan 2012: todo: give size_ambig the 'offset' information
        #        A_x = apply_linear( x, 1 ); % celldisp( size(A_x) )
        #    else
        #        A_x = tfocs_zeros( otp_dims );
        #    end
        # end
        # if np.empty(self.A_x): # i. e. ambiguous size in output dimensions
        #    if identity_linop || zero_x0 && square_linop, # identity_linop??

    def init_iterate_values(self):
        # y values
        self.y = self.x
        self.A_y = self.A_x
        self.C_y = float('inf')
        self.f_y = self.f_x
        self.g_y = self.g_x
        self.g_Ay = self.g_Ax
        # norm_x?

        # z values
        self.z = self.x
        self.A_z = self.A_x
        self.C_z = self.C_x
        self.f_z = self.f_x
        self.g_z = self.g_x
        self.g_Az = self.g_Ax

    # only use on first initialization
    def init_x(self, x):
        # tfocs uses way more checks
        self.x = x

    def reset_yz(self):
        self.y = self.x
        self.A_y = self.A_x
        self.f_y = self.f_x
        self.g_y = self.g_x
        self.g_Ay = self.g_Ax
        self.C_y = self.C_x

        self.z = self.x
        self.A_z = self.A_x
        self.f_z = self.f_x
        self.g_z = self.g_x
        self.g_Az = self.g_Ax
        self.C_z = self.C_x
