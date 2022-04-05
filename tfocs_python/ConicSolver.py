import numpy as np

class ConicSolver:

    def __init__(self):
        # instance attributes taken from tfocs_initialize.m
        self.max_iterations = float('inf')
        self.max_counts = float('inf')
        self.count_ops = False
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
        self.mu = 0
        self.fid = 1
        self.stop_criterion = 1
        self.alg = 'AT'
        self.restart = float('inf')
        self.print_stop_criterion = False
        self.counter_reset = -50
        self.cg_restart = float('inf')
        self.cg_type = 'pr'
        self.stop_criterion_always_use_x = False
        self.data_collection_always_use_x = False
        self.output_always_use_x = False
        self.auto_restart = 'gra' # function or gradient
        self.print_restart = True
        self.debug = False


    def auslender_teboulle(self, smooth_func, affine_func, projector_func, x0):
        """Auslender & Teboulle's method
        args:
            smooth_func: function for smooth

        """
        alg = 'AT'

        # following taken from tfocs_initialize.m
        L = self.L_0
        theta = float('inf')
        f_v_old = float('inf')
        x = []
        A_x = []
        f_x = float('inf')
        C_x = float('inf')
        g_x = []
        g_Ax = []
        restart_iter = 0
        warning_lipschitz = 0
        backtrack_simple = True
        backtrack_tol = 1e-10
        backtrack_steps = 0

        # iteration values
        y = x
        z = x
        A_y = A_x
        A_z = A_x
        C_y = float('inf')
        C_z = C_x
        f_y = f_x
        f_z = f_x
        g_y = g_x
        g_z = g_x
        g_Ay = g_Ax
        g_Az = g_Ax

        while True:
            x_old = x
            z_old = z
            A_x_old = A_x
            A_z_old = A_z

            # backtracking loop
            L_old = L
            L = L * self.alpha
            theta_old = theta

            while True:
                # acceleration

                # what is this?
                theta = advance_theta(theta_old, L, L_old)

    def advance_theta(self):
        None


