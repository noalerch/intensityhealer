import math
import numpy as np

class ConicSolver:
    def __init__(self) -> None:
        # instance attributes taken from tfocs_initialize.m
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
        self.L_local = 0 # changes with backtrack
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
        self.auto_restart = 'gra' # function or gradient
        self.print_restart = True
        self.debug = False

        # iterations start at 0
        self.n_iter = 0

        # TODO: implement out in a smart way
        self.out = np.array([])
        self.test = []

        # TODO: description

        # TODO: function types assertions?

        # TODO: L0_default, alpha default etc
        # def_fields?

        # TODO: some more stuff

        # TODO: smooth & projector function

        # TODO: max min

        # TODO: affine

        # TODO: init tfocs_count___ variable here (found in self.count)
        self.L = self.L_0
        self.theta = float('inf')
        f_v_old = float('inf')
        self.x = np.array([])
        self.A_x = np.array([])
        self.f_x = float('inf')
        self.C_x = float('inf')
        self.C_y = float('inf')
        self.g_x = np.array([])
        self.g_Ax = np.array([])

        self.restart_iter = 0
        self.warning_lipschitz = 0
        self.backtrack_simple = True
        self.backtrack_tol = 1e-10
        self.backtrack_steps = 0

        self.just_restarted = False

    def auslender_teboulle(self, smooth_func, affine_func, projector_func, linear_func, x0):
        """Auslender & Teboulle's method
        args:
            smooth_func: function for smooth

        """
        alg = 'AT'

        # following taken from tfocs_initialize.m
        L = self.L_0
        theta = float('inf')
        f_v_old = float('inf')

        # TODO: investigate if empty lists should be numpy arrays instead
        # x = [] # FIXME: taken from matlab (TFOCS), should probably be number

        counter_Ay = 0
        counter_Ax = 0

        # iteration values
        # init iteration values (tfocs_initialize.m, lines 582-8)
        # it is somewhat more comfortable if they are here rather than in __init__
        # TODO: move most of these, they do not need to be in this method
        y = self.x
        z = self.x
        A_y = self.A_x
        A_z = self.A_x

        while True:
            x_old = self.x
            z_old = z
            A_x_old = self.A_x
            A_z_old = A_z

            # backtracking loop
            L_old = self.L
            self.L = self.L * self.alpha
            theta_old = theta

            # FIXME: theta is Inf
            while True:
                # acceleration
                theta = self.advance_theta(theta_old, L, L_old)

                # next iteration
                if theta < 1:
                    y = (1 - theta) * x_old + theta * z_old

                    if counter_Ay >= self.counter_reset:
                        # A_y = self.apply_linear(y, 1)
                        A_y = linear_func(y, 1) #, mode) # ignoring mode for now

                        counter_Ay = 0

                    else:
                        counter_Ay += 1
                        A_y = (1 - theta) * A_x_old + theta * A_z_old

                f_y = float('inf')
                g_Ay = np.array([]) # should be numpy array?
                g_y = np.array([]) # see above

                if g_y.size == 0:
                    if g_Ay.size == 0:

                        # syntax makes no sense
                        # np.array([f_y, g_Ay]) = self.apply_smooth(A_y)
                        # (f_y, g_Ay) = self.apply_smooth(A_y)
                        # assume for now that count_ops = 1.
                        # in TFOCS,
                        # apply_smooth = @(x)solver_apply(1: (1 + (nargoutt > 1)), smoothF, x );
                        # we just perform the smooth function directly

                        g_y = linear_func(g_Ay, 2)
                        # g_y = self.apply_linear(g_Ay, 2)

                step = 1 / (theta * L)

                # FIXME: i do not understand this. moving on for now
                # np.array[C_z, z] = projector_function(z_old - step * g_y, step)
                C_z, z = projector_func(z_old - step * g_y, step)
                A_z = linear_func(z, 1)

                # new iteration
                if theta == 1:
                    x = z
                    A_x = A_z
                    C_x = C_z

                else:
                    x = (1 - theta) * x_old + theta * z

                    if counter_Ax >= self.counter_reset:
                        counter_Ax = 0
                        A_x = linear_func(x, 1)
                    else:
                        counter_Ax += 1
                        A_x = (1 - theta) * A_x_old + theta * A_z

                    C_x = float('inf')

                f_x = float('inf')

                # TODO: should these be numpy arrays?
                g_Ax = np.array([])
                g_x = np.array([])

                # TODO: investigate further the use of do_break
                #       in tfocs_AT. is it necessary with the
                #       changed backtrack?
                break_val = self.backtrack(self.x, y, f_y, g_x, g_y, A_y, g_Ax, g_Ay, smooth_func)
                if break_val:
                    break

            # TODO: proper implementation of xy if we want to handle
            #       stopCrit 2
            break_val = self.iterate(x, y, x_old, A_y, f_y)
            if break_val:
                break

        self.cleanup()

    # no idea what this method should do rofl
    def cleanup(self):
        pass

    # based on tfocs_iterate.m script
    def iterate(self, x, y, x_old, A_y, f_y,
                smooth_function, projector_function) -> bool:
        status = ""

        # test for positive stopping criteria
        new_iter = self.n_iter + 1
        norm_x = np.linalg.norm(x)
        norm_dx = np.linalg.norm(x - x_old)

        xy_sq = 0 # placeholder

        # legacy stopping criteria
        # not necessary for jackdaw-based COACS
        if self.stop_criterion == 2 and self.beta >= 1:
            xy = x - y

            xy_sq = square_norm(xy)

        current_dual = None

        limit_reached = False # bool nicer than the string search in tfocs

        # could perhaps use match-case which was introduced in Python 3.10
        # avoiding this due to compatibility issues
        if np.isnan(f_y):
            status = "NaN found -- aborting"
        elif self.stop_criterion == 1 and norm_dx == 0:
            if self.n_iter > 1:
                status = "Step size tolerance reached (||dx||=0)"
        elif self.stop_criterion == 1 and norm_dx < self.tol * max(norm_x, 1):
            status = "Step size tolerance reached"

        elif self.stop_criterion == 2 and self.L * math.sqrt(xy_sq) < self.tol * max(norm_x, 1):
            status = "Step size tolerance reached"
        elif self.n_iter == self.max_iterations:
            status = "Iteration limit reached"
            limit_reached = True
        elif self.count_ops and np.max(self.count) <= self.max_counts:
            status = "Function/operator count limit reached"
            limit_reached = True
        elif self.backtrack_steps > 0 and xy_sq == 0:
            status = f"Unexpectedly small step size after {self.backtrack_steps} backtrack steps"

        # for stop_crit 3, need new and old dual points
        # TODO most of this. Left for now because not needed for COACS
        if self.stop_criterion == 3 or self.stop_criterion == 4:
            if not self.saddle:
                raise "stop criterion {3, 4} requires a saddle point problem"


        ### Use function value for y instead of x if cheaper
        ### This part determines computational cost before continuing
        v_is_x = False
        v_is_y = False

        # Honestly unsure if any of these conditions will ever be true in COACS
        if (status == "" or limit_reached) and (self.stop_function is not None
                or self.restart < 0 or self.stop_criterion in [3, 4]):
            need_dual = self.saddle and (self.stop_function is None or
                                         self.stop_criterion in [3, 4])

            # unsure of these tfocs_iterate.m lines 60-1
            comp_x = [np.isinf(self.f_x), need_dual * np.isempty(self.g_Ax), np.isinf(self.C_x)]
            comp_y = [np.isinf(f_y), need_dual * np.isempty(self.g_Ay), np.isinf(self.C_y)]

            if np.sum(comp_x) <= np.sum(comp_y) or self.stop_criteria_always_use_x:

                if comp_x[2]:
                    self.f_x, self.g_Ax = smooth_function(self.A_x)
                elif comp_x[1]:
                    f_x = smooth_function(self.A_x)

                current_priority = x
                if self.saddle:
                    current_dual = self.g_Ax
                f_v = np.maxmin(self.f_x + self.C_x)
                v_is_x = True

            else:

                if comp_y[2]:
                    f_y, g_Ay = smooth_function(A_y)
                elif comp_y[1]:
                    f_y = smooth_function(A_y)

                current_priority = y
                if self.saddle:
                    current_dual = g_Ay
                f_v = np.maxmin(f_y + self.C_y)
                v_is_y = True

                if self.data_collection_always_use_x:
                    f_vy = f_v
                    if self.saddle:
                        dual_y = current_dual

            # TODO: llnes 84-96 in tfocs_iterate.m
            #       likely unnecessary for COACS
            print("Unexpected! Please implement lines 84 from tfocs_iterate.m")



        #    # TODO: finish this part
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

                f_x_save = self.f_x
                g_Ax_save = self.g_Ax

                if self.error_function is not None and self.saddle:

                    if self.g_Ax is not None:
                        self.f_x, self.g_Ax = smooth_function(self.A_x)

                    current_dual = self.g_Ax

                if np.isinf(self.f_x):
                    f_x = smooth_function(self.A_x)

                if np.isinf(self.C_x):
                    C_x = projector_function(x)

                # might be incorrect, if f_X and C_x are arrays
                f_v = self.max_min * (f_x + C_x)
                cur_pri = x # want better name but idk what this means
                v_is_x = True
                # Undo calculations
                self.f_x = f_x_save
                self.g_Ax = g_Ax_save

            # if ~isempty(errFcn) & & iscell(errFcn)
            # python has no cell array (most like Python list)
            # what to do here?
            # TODO: ignoring this case for now
            #       please investigate but it does not seem error_function in this impl
            #       will ever be a matlab cell array equivalent...
            #if self.error_function is not None and np.iscell(self.error_function):
            #    errs = np.zeros(1, )

            # if ~isempty(stopFcn)
            # again irrelevant for jackdaw COACS. TODO


        # iterate line 226
        if status == "" and self.beta < 1 and self.backtrack_simple \
                             and self.L_local > self.L_exact:
            # NOTE: it appears localL in TFOCS arises from the backtracking logic
            # we put L_local as a class instance attribute
            warning_lipschitz = True
        # else probably not needed
        # else:
            # warning_lipschitz = False

        # print status
        if will_print:
            if warning_lipschitz:
                warning_lipschitz = False
                bchar = 'L'

            elif self.backtrack_simple:
                bchar = ' '

            else:
                bchar = '*'

            # TODO: format may be (read: is likely) incorrect
            # TODO: pass f_v and norm_x as params
            to_print ="(%d, '%-4d| %+12.5e  %8.2e  %8.2e%c)" % self.fid, self.n_iter, f_v, norm_dx / max(norm_x, 1), 1 / self.L, {bchar}

            # NOTE: matlab fprintf prints to file!
            #       could perhaps use more elegant write method
            print(to_print, file=self.fid)

            if self.count_ops:
                print("|", file=self.fid)

                # TODO: tfocs_count___ is array??
                print("%5d", self.count, file=self.fid)

            if self.error_function is not None:
                if self.count_ops:
                    print(' ', file=self.fid)

                print('|', file=self.fid)
                # TODO: no errs since error function is null by default
                # print(" {:8.2e}".format(errs))

            # display number used to determine stopping
            # in COACS this should always be 1
            if self.print_stop_criteria:
                if self.stop_criterion == 1:
                    if norm_dx is not None and norm_x is not None:
                        stop_resid = norm_dx/max(norm_x, 1)

                    else:
                        stop_resid = float('inf')

                else:
                    raise Exception(f"stop criterion {self.stop_criterion} not yet implemented")

                if self.error_function is not None or self.count_ops:
                    print(' ', file=self.fid)

                print('|', file=self.fid)

                # assumes stop_resid exists (i. e. stop_criterion == 1)
                print(" %8.2e", stop_resid, file=self.fid) # hopefully correct syntax

            if self.print_restart and self.just_restarted:
                print(' | restarted', file=self.fid)

            print('\n', file=self.fid)

        # extending arrays if needed
        if self.save_history:
            f_size = self.out.f.size
            if f_size < self.n_iter and status == "":
                csize = min(self.max_iterations, f_size + 1000) # this is +1 compated to TFOCS due to matlab indexing. Does this matter?

                # removed + 1 because of 0-indexing
                self.out.f = np.pad(self.out.f, ((0, csize), (0, 0))) # TODO: verify
                self.out.theta = np.pad(self.out.theta, ((0, csize), (0, 0))) # TODO: verify
                self.out.step_size = np.pad(self.out.step_size, ((0, csize), (0, 0))) # TODO: verify
                self.out.norm_grad = np.pad(self.out.norm_grad, ((0, csize), (0, 0))) # TODO: verify

                if self.count_ops:

                    # uses : instad of 1 in matlab code. Please check!
                    self.out.norm_grad = np.pad(self.out.norm_grad, ((0, csize), (0, 0)))  # TODO: verify

                # TODO: check indexing


        return True #### TODO TODO





    # based on Nettelblad's changed backtracking logic for TFOCS
    # handles numerical errors better
    def backtrack(self, x, y, f_y, g_x, g_y, A_y, g_Ax, g_Ay, smooth_func):

        if self.beta >= 1:
            return

        xy = x - y

        ## TODO TODO TODO

        # TODO: double check parenthesis
        val = max(abs(xy.flatten()) - np.finfo(max(max(abs(xy.flatten())), max(abs(x.flatten()), abs(y.flatten())))))
        xy_sq = square_norm(val)  # TODO: correct square norm?
        ## TODO TODO TODO

        if xy_sq == 0:
            self.L_local = float('inf')
            return


        # to handle numerical issues from the ratio being smaller than machine epsilon
        if xy_sq / (square_norm(x)) < np.finfo(float).eps:
            self.counter_Ax = float('inf')

        if self.g_Ax.size == 0 or np.isinf(self.f_x):
            self.f_x, self.g_Ax = smooth_func(self.A_x)

        # not sure what to call this temp variable
        # in tfocs_backtrack it simply overwrites backtrack_simple
        # before changing again in the next lines
        within_tolerance = abs(f_y - self.f_x) >=\
                                self.backtrack_tol * max(max(abs(self.f_x),
                                                             abs(f_y)), 1)

        # .^ is in matlab elementwise power, we represent as **
        self.backtrack_simple = within_tolerance and (abs(xy_sq) >= self.backtrack_tol**2)

        # assuming np.dot is equivalent to tfocs_dot
        L_local_origin = 2 * np.dot(self.A_x - A_y, g_Ax - g_Ay) / xy_sq

        self.L_local = max(self.L, L_local_origin)

        q_x = np.dot(xy, g_y + 0.5 * self.L * xy)

        L_local_2 = self.L + 2 * max((self.f_x - f_y) - q_x + max([np.finfo(float).eps(self.f_x), np.finfo(float).eps(f_y), np.finfo(float).eps(q_x), np.finfo(float).eps(self.f_x - f_y)]), 0) / xy_sq

        if self.backtrack_simple:
            self.L_local = min(self.L_local, L_local_2)

        # NOTE: that normlimit in nettelblads backtrack is only called from
        #       code which is already commented out
        # norm_limit = np.array([abs(xy_sq) / (self.backtrack_tol * max(max(abs(np.dot(x, x)), abs(np.dot(y, y))), 1)))])

        self.backtrack_steps += 1

        if self.f_x - f_y > 0:
            self.L_local = max(self.L, self.L_local)

        if self.L_local <= self.L or self.L_local >= self.L_exact:
            return # analogous to break in matlab script?

        # isinf would be strange here since self.L_local should be a number
        # TODO: check other isinfs
        # if np.isinf(self.L_local)
        if self.L_local == float('inf'):
            self.L_local = self.L

        self.L = min(self.L_exact, self.L / self.beta)





    # assuming countOps (?), see tfocs_initialize.m line 398
    # TODO: remove varargin?
    def apply_projector(self, varargin, projector_function):
        if self.count_ops:
            pass

        # false by default
        else:
            return projector_function(varargin)


    # TODO? ignore for now
    def apply_linear(self, x, mode):
        pass
        # this can't be right lol
        # return self.solver_apply(3, self.linear_function, x, mode)

    # TODO
    def solver_apply(self):
        pass

    # TODO
    def linear_function(self):
        pass

    def test_method(self):
        return "task tested successfully"


    # assumes mu > 0 & & ~isinf(Lexact) && Lexact > mu,
    # see tfocs_initialize.m (line 532-) and healernoninv.m
    def advance_theta(self, theta_old: float, L, L_old):
        # TODO: N83 check? probably don't need to worry about this
        # TODO: warning that AT may give wrong results with mu > 0 ?
        # TODO: calculating this inside theta expensive. move outside?
        ratio = math.sqrt(self.mu / self.L_exact)
        theta_scale = (1 - ratio) / (1 + ratio)
        return min(1.0, theta_old * theta_scale)

class SolverOutput:
    def __init__(self, alg, f):
        self.alg = alg
        self.f = f
        self.theta = np.array([])
        self.step_size = np.array([])
        self.norm_grad = np.array([])

def square_norm(arr):
    return math.sqrt(np.dot(arr, arr))
