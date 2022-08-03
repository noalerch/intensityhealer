import math
import numpy as np
from ConicInternal import projection_Rn


def square_norm(arr):
    return math.sqrt(np.dot(arr, arr))

# TODO: affine_func, projector_func are optional
class ConicSolver:
    def __init__(self, smooth_func, affine_func, projector_func, x0) -> None:
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


        # TODO: max min

        # TODO: affine

        # TODO: init tfocs_count___ variable here (found in self.count)
        self.L = self.L_0
        self.theta = float('inf')
        self.f_v_old = float('inf')

        self.f_v = None  # i don't know

        self.iv = IterationVariables
        # let x = x0 unconditionally for now
        self.iv.x = x0

        self.restart_iter = 0
        self.warning_lipschitz = False  # 0 in matlab
        self.backtrack_simple = True
        self.backtrack_tol = 1e-10
        self.backtrack_steps = 0

        self.just_restarted = False

        self.output = None

        self.apply_linear = None

        # the way this works in TFOCS is that affineF
        # is a cell array of an arbitrary amount of
        # linear functions each paired with an offset
        self.affine_handler(affine_func)


        #self.set_linear(linear_func, offset)
        #print(self.apply_linear)

        if self.adjoint:
            pass  # TODO


        # self.set_linear(linear_func)

        # TODO: should we support multiple smooth functions
        self.apply_smooth = None  # ?
        self.set_smooth(smooth_func)

        # self.apply_projector = None
        self.set_projector(projector_func)

        # TODO: check function types

        # TODO: pass through projector

        # TODO: get linear function from affine
        #       linop_stack



    def solve(self):
        """

        assumes Auslender-Teboulle algorithm for now
        """
        iv = IterationVariables()

        self.auslender_teboulle(iv)

        return self.output

    def auslender_teboulle(self, iv):
        """Auslender & Teboulle's method
        args:
            smooth_func: function for smooth
            affine_func: function for smooth

        """

        self.output = SolverOutput('AT')

        # following taken from tfocs_initialize.m
        L = self.L_0
        # theta = float('inf')
        # self.f_v_old = float('inf')

        counter_Ay = 0
        counter_Ax = 0

        # iteration values
        # init iteration values (tfocs_initialize.m, lines 582-8)
        # it is somewhat more comfortable if they are here rather than in __init__

        while True:
            x_old = iv.x
            z_old = iv.z
            A_x_old = iv.A_x
            A_z_old = iv.A_z

            # backtracking loop
            L_old = self.L
            self.L = self.L * self.alpha
            theta_old = self.theta

            # FIXME: theta is Inf
            while True:
                # acceleration
                self.theta = self.advance_theta(theta_old)

                # next iteration
                if self.theta < 1:
                    iv.y = (1 - self.theta) * x_old + self.theta * z_old

                    if counter_Ay >= self.counter_reset:
                        # A_y = self.apply_linear(y, 1)
                        iv.A_y = self.apply_linear(iv.y, 2)

                        counter_Ay = 0

                    else:
                        counter_Ay += 1
                        A_y = (1 - self.theta) * A_x_old + self.theta * A_z_old

                iv.f_y = float('inf')
                iv.g_Ay = np.array([])
                iv.g_y = np.array([])

                if iv.g_y.size == 0:
                    if iv.g_Ay.size == 0:

                        # syntax makes no sense
                        # np.array([f_y, g_Ay]) = self.apply_smooth(A_y)
                        # (f_y, g_Ay) = self.apply_smooth(A_y)
                        # assume for now that count_ops = 1.
                        # in TFOCS,
                        # apply_smooth = @(x)solver_apply(1: (1 + (nargoutt > 1)), smoothF, x );
                        # we just perform the smooth function directly

                        iv.g_y = self.apply_linear(iv.g_Ay, 2)
                        # g_y = self.apply_linear(g_Ay, 2)

                step = 1 / (self.theta * L)

                # np.array[C_z, z] = projector_function(z_old - step * g_y, step)
                print(self.apply_projector)
                # FIXME: apply_projector is None and cannot be called
                iv.C_z, iv.z = self.apply_projector(z_old - step * iv.g_y, step, grad=1)
                iv.A_z = self.apply_linear(iv.z, 1)

                # new iteration
                if self.theta == 1:
                    iv.x = iv.z

                    iv.A_x = iv.A_z
                    iv.C_x = iv.C_z

                else:
                    iv.x = (1 - self.theta) * x_old + self.theta * iv.z

                    if counter_Ax >= self.counter_reset:
                        counter_Ax = 0
                        iv.A_x = self.apply_linear(iv.x, 1)
                    else:
                        counter_Ax += 1
                        iv.A_x = (1 - self.theta) * A_x_old + self.theta * iv.A_z

                    iv.C_x = float('inf')

                iv.f_x = float('inf')

                iv.g_Ax = np.array([])
                iv.g_x = np.array([])

                break_val = self.backtrack(iv, counter_Ax)
                if break_val:
                    break

            # TODO: proper implementation of xy if we want to handle
            #       stopCrit 2
            break_val, v_is_x, v_is_y, f_vy, status = self.iterate(iv, x_old, A_x_old)
            if break_val:
                break

        self.cleanup(v_is_x, v_is_y, f_vy, self.iv, status)

    def cleanup(self, v_is_x, v_is_y, f_vy, iv, status):
        # TODO: cur_dual (probably not needed for COACS)
        n_iter = self.n_iter

        if v_is_y and not self.output_always_use_x \
                and not self.data_collection_always_use_x:

            f_vy = self.f_v
        #    if self.saddle:
        #        dual_y = cur_dual

        if not v_is_x:
            if self.saddle:
                if iv.g_Ax is None:
                    iv.f_x, iv.g_Ax = self.apply_smooth(iv.A_x, grad=1)

                # cur_dual = get_dual(self.g_Ax)

            elif np.isinf(iv.C_x):
                iv.C_x = self.apply_projector(iv.x)

            self.f_v = self.max_min * (iv.f_x + iv.C_x)
            cur_pri = iv.x

        # take whichever of x or y is better
        x_or_y_string = 'x'
        if v_is_y and not self.output_always_use_x and f_vy < self.f_v:
            self.f_v = f_vy

            #if self.saddle:
                #cur_dual = dual_y

            iv.x = iv.y
            x_or_y_string = 'y'

        # ignoring because not saddle by default in tfocs
        # if self.saddle:

        if self.fid and self.print_every:
            print("Finished: %s\n" % status)  # , file=self.fid)

        self.output.n_iter = self.n_iter
        self.output.status = status
        self.output.x_or_y = x_or_y_string

        # TODO: d description var

        if self.save_history:
            # where does f come from?
            self.output.f[n_iter - 1] = self.f_v  # TODO: matlab vs python indexing?

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
    def iterate(self, iv, A_x_old, x_old):
        status = ""

        v_is_x = False
        v_is_y = False

        # this kind of defeats the purpose of data_collection_always_use_x
        # but this is a simple solution
        # NOTE: this MIGHT lead to bugs!
        f_vy = self.f_v

        # test for positive stopping criteria
        new_iter = self.n_iter + 1
        norm_x = np.linalg.norm(iv.x)
        norm_dx = np.linalg.norm(iv.x - x_old)

        xy_sq = 0  # placeholder

        # legacy stopping criteria
        # not necessary for jackdaw-based COACS
        if self.stop_criterion == 2 and self.beta >= 1:
            xy = iv.x - iv.y

            xy_sq = square_norm(xy)

        current_dual = None

        limit_reached = False # bool nicer than the string search in tfocs

        # could perhaps use match-case which was introduced in Python 3.10
        # avoiding this due to compatibility issues
        if np.isnan(iv.f_y):
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
            comp_x = np.array([np.isinf(iv.f_x), need_dual * np.isempty(iv.g_Ax), np.isinf(iv.C_x)])
            comp_y = np.array([np.isinf(iv.f_y), need_dual * np.isempty(iv.g_Ay), np.isinf(iv.C_y)])

            if np.sum(comp_x) <= np.sum(comp_y) or self.stop_criteria_always_use_x:

                if comp_x[2]:
                    iv.f_x, iv.g_Ax = self.apply_smooth(iv.A_x, grad=1)
                elif comp_x[1]:
                    iv.f_x = self.apply_smooth(iv.A_x, grad=1)

                current_priority = iv.x
                if self.saddle:
                    current_dual = iv.g_Ax
                self.f_v = np.maxmin(iv.f_x + iv.C_x)
                v_is_x = True

            else:

                if comp_y[2]:
                    iv.f_y, iv.g_Ay = self.apply_smooth(iv.A_y, grad=1)
                elif comp_y[1]:
                    iv.f_y = self.apply_smooth(iv.A_y, grad=1)

                current_priority = iv.y
                if self.saddle:
                    current_dual = iv.g_Ay

                self.f_v = np.maxmin(iv.f_y + iv.C_y)
                v_is_y = True

                if self.data_collection_always_use_x:
                    f_vy = self.f_v
                    if self.saddle:
                        dual_y = current_dual

            # TODO: llnes 84-96 in tfocs_iterate.m
            #       likely unnecessary for COACS
            raise RuntimeWarning("Unexpected! Please implement lines 84 from tfocs_iterate.m")



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

                f_x_save = iv.f_x
                g_Ax_save = iv.g_Ax

                # both of these should be false for COACS
                if self.error_function is not None and self.saddle:

                    if iv.g_Ax is not None:
                        iv.f_x, iv.g_Ax = self.apply_smooth(iv.A_x, grad=1)

                    current_dual = iv.g_Ax

                if np.isinf(iv.f_x):
                    iv.f_x = self.apply_smooth(iv.A_x, grad=1)

                if np.isinf(iv.C_x):
                    iv.C_x = self.apply_projector(iv.x)

                self.f_v = self.max_min * (iv.f_x + iv.C_x)
                cur_pri = iv.x # want better name but idk what this means
                v_is_x = True
                # Undo calculations
                iv.f_x = f_x_save
                iv.g_Ax = g_Ax_save

            # if ~isempty(errFcn) & & iscell(errFcn)
            # python has no cell array (most like Python list)
            # what to do here?
            # TODO: ignoring this case for now
            #       please investigate but it does not seem error_function in this impl
            #       will ever be a matlab cell array equivalent...
            #       (also, it is null in COACS)
            #if self.error_function is not None and np.iscell(self.error_function):
            #    errs = np.zeros(1, )

            # if ~isempty(stopFcn)
            # again irrelevant for jackdaw COACS. TODO


        # iterate line 226
        if status == "" and self.beta < 1 and self.backtrack_simple \
                and self.L_local > self.L_exact:
            # NOTE: it appears localL in TFOCS arises from the backtracking logic
            # we put L_local as a class instance attribute
            self.warning_lipschitz = True
        # else probably not needed
        # else:
            # warning_lipschitz = False

        # print status
        if will_print:
            if self.warning_lipschitz:
                self.warning_lipschitz = False
                bchar = 'L'

            elif self.backtrack_simple:
                bchar = ' '

            else:
                bchar = '*'

            # TODO: format may be incorrect
            to_print = "(%d, '%-4d| %+12.5e  %8.2e  %8.2e%c)" % (self.fid, self.n_iter, self.f_v,
                    norm_dx / max(norm_x, 1), 1 / self.L, {bchar})

            # NOTE: matlab fprintf prints to file!
            # TODO: all prints are for now just to stdout
            print(to_print) #, file=self.fid)

            if self.count_ops:
                print("|") # , file=self.fid)

                # TODO: tfocs_count___ is array??
                print("%5d", self.count) #, file=self.fid)

            if self.error_function is not None:
                if self.count_ops:
                    print(' ') #, file=self.fid)

                print('|') #, file=self.fid)
                # TODO: no errs since error function is null by default
                #       thus, ignore for now
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
                    print(' ')  # , file=self.fid)

                print('|')  #, file=self.fid)

                # assumes stop_resid exists (i. e. stop_criterion == 1)
                print(" %8.2e", stop_resid)  # , file=self.fid) # hopefully correct syntax

            if self.print_restart and self.just_restarted:
                print(' | restarted')  #, file=self.fid)

            print('\n') # , file=self.fid)

        # extending arrays if needed
        if self.save_history:
            f_size = self.out.f.size
            if f_size < self.n_iter and status == "":
                csize = min(self.max_iterations, f_size + 1000)  # this is +1 compated to TFOCS due to matlab indexing. Does this matter?

                # removed + 1 because of 0-indexing
                self.out.f = np.pad(self.out.f, ((0, csize), (0, 0)))  # TODO: verify
                self.out.theta = np.pad(self.out.theta, ((0, csize), (0, 0)))  # TODO: verify
                self.out.step_size = np.pad(self.out.step_size, ((0, csize), (0, 0)))  # TODO: verify
                self.out.norm_grad = np.pad(self.out.norm_grad, ((0, csize), (0, 0)))  # TODO: verify

                if self.count_ops:

                    # uses : instad of 1 in matlab code. Please check!
                    self.out.norm_grad = np.pad(self.out.norm_grad, ((0, csize), (0, 0)))  # TODO: verify

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
                do_auto_restart = np.dot(iv.g_Ay, iv.A_x - A_x_old)

            elif self.auto_restart == 'fun':
                do_auto_restart = self.max_min * iv.f_v > self.max_min * self.f_v_old

        if self.n_iter - self.restart_iter == abs(round(self.restart)) \
                or do_auto_restart:

            self.restart_iter = self.n_iter
            self.backtrack_simple = True
            self.theta = float('inf')

            iv.reset_yz()

            self.f_v_old = self.max_min * float('inf')

            self.just_restarted = True

        # do_auto_restart

        return do_break, v_is_x, v_is_y, f_vy, status

    # based on Nettelblad's changed backtracking logic for TFOCS
    # handles numerical errors better
    def backtrack(self, iv, counter_Ax):

        # instead of setting a do_break variable (which
        # is always done in the context of a break/return
        # in tfocs for matlab version >= R2015b
        # we simply return True and account for this in the
        # call to this function
        if self.beta >= 1:
            return True, counter_Ax

        xy = iv.x - iv.y

        # TODO: double check parenthesis
        print(iv.x)
        print(iv.y)

        print(xy.flatten())
        print(xy)

        # is flatten even necessary?
        val = max(abs(xy.flatten()) -
                    np.finfo(max(max(abs(xy.flatten())),
                        max(abs(iv.x.flatten()), abs(iv.y.flatten())))))
        xy_sq = square_norm(val)

        if xy_sq == 0:
            self.L_local = float('inf')
            return True, counter_Ax

        # to handle numerical issues from the ratio being smaller than machine epsilon
        # force reset
        if xy_sq / (square_norm(iv.x)) < np.finfo(float).eps:
            counter_Ax = float('inf')
            return True, counter_Ax

        if iv.g_Ax.size == 0 or np.isinf(iv.f_x):
            iv.f_x, iv.g_Ax = self.apply_smooth(iv.A_x, grad=1)

        # in tfocs_backtrack it simply overwrites backtrack_simple
        # before changing again in the next lines
        within_tolerance = abs(iv.f_y - iv.f_x) >= \
            self.backtrack_tol * max(max(abs(iv.f_x), abs(iv.f_y)), 1)

        # .^ is in matlab elementwise power, we represent as **
        self.backtrack_simple = within_tolerance and (abs(xy_sq) >= self.backtrack_tol**2)

        # assuming np.dot is equivalent to tfocs_dot
        L_local_origin = 2 * np.dot(iv.A_x - iv.A_y, iv.g_Ax - iv.g_Ay) / xy_sq

        self.L_local = max(self.L, L_local_origin)

        q_x = np.dot(xy, iv.g_y + 0.5 * self.L * xy)

        L_local_2 = self.L + 2 * max((iv.f_x - iv.f_y) - q_x + max([np.finfo(float).eps(iv.f_x), np.finfo(float).eps(iv.f_y), np.finfo(float).eps(q_x), np.finfo(float).eps(iv.f_x - iv.f_y)]), 0) / xy_sq

        if self.backtrack_simple:
            self.L_local = min(self.L_local, L_local_2)

        # NOTE: that normlimit in nettelblads backtrack is only called from
        #       code which is already commented out
        # norm_limit = np.array([abs(xy_sq) / (self.backtrack_tol * max(max(abs(np.dot(x, x)), abs(np.dot(y, y))), 1)))])

        self.backtrack_steps += 1

        if iv.f_x - iv.f_y > 0:
            self.L_local = max(self.L, self.L_local)

        if self.L_local <= self.L or self.L_local >= self.L_exact:
            return True, counter_Ax # analogous to break in matlab script?

        # if np.isinf(self.L_local):
        #    pass
        elif self.L_local == float('inf'):
            self.L_local = self.L

        self.L = min(self.L_exact, self.L / self.beta)

        return False, counter_Ax


    # TODO? ignore for now
    #def apply_linear(self, mode):
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

    def set_linear(self, linear_func, offset):
        if self.count_ops:
            # hack to allow multiple return values
            # pass grad = 1 to return gradient as well
            self.apply_linear = lambda x, mode, grad = 0: self.solver_apply(2, linear_func, [x, mode, grad])
        else:
            self.apply_linear = lambda x, mode, grad = 0: linear_func(x, mode, grad)

    def affine_handler(self, affine):
        """

        :param affine: list of pairs of linear functions and offsets (affine)
        :return:
        """
        # TODO: fix input/output dimensions from projector and smooth
        if affine is None:
            # TODO: probably incomplete identity. for example, handle gradient?
            self.apply_linear = lambda x, grad=0: x
        else:
            pass

    def set_smooth(self, smooth_func):
        if self.count_ops:
            # TODO: first argument to solver_apply is strange
            self.apply_smooth = lambda x, grad = 0: self.solver_apply([i for i in range(0, 1 + grad)],
                                                                      smooth_func, [x, grad])
        else:
            self.apply_smooth = smooth_func

    def set_projector(self, projector_func):
        if projector_func is None:
            # if isempty(projectorF),
            # n_proj = 0 # when is this used?
            # projectorF = proj_Rn
            # projector_func = lambda args:

            # based on my understanding of smooth_constant:
            projector = projection_Rn

        else:
            #
            projector = projector_func  # proximity_stack(projector)

        if self.count_ops:
            self.apply_projector = lambda args, grad = 0: self.solver_apply([i for i in range(3, 4 + grad)],
                                                                        projector, args)

        else: self.apply_projector = projector

    # projection onto entire space

    # TODO
    def solver_apply(self, ndxs, func, args):
        self.count[ndxs] += 1
        return func(*args)

    # TODO
    def linear_function(self):
        pass

    def test_method(self):
        return "task tested successfully"

    # assumes mu > 0 & & ~isinf(Lexact) && Lexact > mu,
    # see tfocs_initialize.m (line 532-) and healernoninv.m
    def advance_theta(self, theta_old: float):
        # TODO: N83 check? probably don't need to worry about this
        # TODO: warning that AT may give wrong results with mu > 0 ?
        # TODO: calculating this inside theta expensive. move outside?
        ratio = math.sqrt(self.mu / self.L_exact)
        theta_scale = (1 - ratio) / (1 + ratio)
        return min(1.0, theta_old * theta_scale)


# TODO: finish output stuff
class SolverOutput:
    def __init__(self, alg):
        self.alg = alg
        # self.f = f  # objective function
        self.theta = np.array([])
        self.step_size = np.array([])
        self.norm_grad = np.array([])
        self.x_or_y = ""
        self.dual = None

    def display(self):
        print("Display not implemented!")


class IterationVariables:
    def __init__(self):

        # x values
        self.x = np.array([])
        self.A_x = np.array([])
        self.f_x = float('inf')
        self.C_x = float('inf')
        self.g_x = np.array([])
        self.g_Ax = np.array([])

        # y values
        self.y = self.x
        self.A_y = self.A_x
        self.f_y = self.f_x
        self.g_y = self.g_x
        self.g_Ay = self.g_Ax
        self.C_y = self.C_x

        # z values
        self.z = self.x
        self.A_z = self.A_x
        self.f_z = self.f_x
        self.g_z = self.g_x
        self.g_Az = self.g_Ax
        self.C_z = self.C_x

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


