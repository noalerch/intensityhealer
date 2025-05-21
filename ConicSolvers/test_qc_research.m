function test_qc_research
%% Tests the solvers on a simple constrained quadratic function
%{
    Solve:

    minimize_x  c'x + x'Dx/2
    subject to  ||x||_1 <= 10

    as an example of using TFOCS without the "SCD" interface
    (since the objective is smooth and we can project, there's 
     no need to smooth and solve the dual)

    It's also an example of using 2 objective functions

%}

%% Now, add in constraints

randn( 'state', sum('quadratic test') );
N = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSTRUCT THE TEST PROBLEM %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = randn(N,1);
D = randn(N,N);
D = D * D' + .5*eye(N);
s = svd(D);
x_star = - D \ c;
L = max(s); 
mu = min(s);
if mu < eps, disp('WARNING: may have 0 eigenvalues'); end
f_star = 0.5 * c' * x_star;
n_star = norm( x_star );
x0 = zeros(N,1);

%f       = @(x) c'*x + x'*D*x/2;
%grad_f  = @(x) c + D*x;
%smoothF = @(x) wrapper_objective( f, grad_f, x );
%   -- or --
%smoothF = smooth_quad(D,c);
%   -- or (demonstrating how to use 2 objective functions) --
f1p      = @(x) c'*x;
f2p      = @(x) x'*D*x/2;
g1p      = @(x) c;
g2p      = @(x) D*x;
smoothFp = { @(x) wrapper_objective(f1p,g1p,x); @(x) wrapper_objective(f2p,g2p,x) };
f1x      = @(x) c'*x + x'*D*x/2;
g1x      = @(x) c + D*x;
smoothFx = @(x) wrapper_objective(f1x,g1x,x);
linearF = { 1 ; 1 };


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET UP THE TEST PARAMETERS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts = [];
opts.tol        = 1e-16;
opts.maxits     = 3000;
opts.restart    = 100;

%%%%%%%%%%%%%%%%%%%%%
% ADD IN CONSTRAINT %
%%%%%%%%%%%%%%%%%%%%%
% projectorF      = proj_simplex(10); % x >=0, sum(x) = 10
projectorF      = proj_l1_edit(10);  % l1 ball, radius 10 (e.g. norm(x,1) <= 10)


%%%%%%%%%%%%%%%%%
% RUN THE TESTS %
%%%%%%%%%%%%%%%%%
[ x, out, optsOut ] = tfocs( smoothFx, linearF, projectorF, x0, opts );
% Check that we are within allowable bounds



function [v,gr] = wrapper_objective(f,g,x)
v = f(x);
if nargout > 1
    gr = g(x);
end
% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.
