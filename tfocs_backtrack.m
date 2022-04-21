% TFOCS_BACKTRACK
% Backtracking helper script.

% Quick exit for no backtracking
do_break = false; % >R2015b compatibility
if beta >= 1
    do_break = true;
    return; % break changed to return for compatability
end % SRB changing == to >=

% Quick exit if no progress made
xy = x - y;
xy_sq = tfocs_normsq( max(abs(xy(:)) - eps(max(max(abs(xy(:))), max(abs(x(:)), abs(y(:))))), 0));
%xy_sq = tfocs_normsq( xy ) * (0.90);
if xy_sq == 0
%  localL = Lexact;
%  if ~isinf(localL), xy_sq = eps(xy_sq); end
  localL = Inf;
  do_break = true;
  return;
end
%fprintf('xy_sq/x is %.2e\n', xy_sq/tfocs_normsq(x) );
if xy_sq/tfocs_normsq(x) < eps, cntr_Ax=Inf; do_break=true; return; end % force a reset

% Compute Lipschitz estimate
if isempty( g_Ax ) || isinf( f_x ),
    [ f_x, g_Ax ] = apply_smooth( A_x );
end

backtrack_simple = abs( f_y - f_x ) >= backtrack_tol * max(max( abs( f_x ), abs( f_y ) ), 1) ;

%backtrack_simple = backtrack_simple && (abs( tcos_normsq(g_x - g_y) ) >= backtrack_tol.^2 * max( abs(tfocs_normsq(g_x)), abs(tfocs_normsq(g_y)) ) );
backtrack_simple = backtrack_simple && (abs( xy_sq ) >= backtrack_tol.^2 * max(max( abs(tfocs_normsq(x)), abs(tfocs_normsq(y))),1));

localLorig = 2 * tfocs_dot( A_x - A_y, g_Ax - g_Ay ) / xy_sq;
%localL = max(L, 2 * max(abs(tfocs_dot( A_x - A_y, g_Ax - g_Ay )), abs(tfocs_dot(A_x - A_y, g_Ax + g_Ay))) / xy_sq);
%localL = 1e60;
%localL = max(L, 2 * abs(tfocs_dot( A_x - A_y, g_Ax - g_Ay )*1.01) / xy_sq);
% g_x = apply_linear(g_Ax, 2);
%localL = max(L, 2 * abs(tfocs_dot( x - y, g_x - g_y )) / xy_sq);

AxAy = abs(A_x - A_y);
%AxAy = AxAy + max(max(eps(AxAy),eps(A_x)), eps(A_y)) * 1e7;
gAxgAy = abs(g_Ax - g_Ay);
%gAxgAy = gAxgAy + max(max(eps(gAxgAy),eps(g_Ax)), eps(g_Ay)) * 1e7;
%localLorig = 2 * ((tfocs_dot( AxAy, gAxgAy))) / xy_sq;
%localLorig = 2 * ((tfocs_dot( AxAy, max(abs(g_Ax), abs(g_Ay)) - min(abs(g_Ax), abs(g_Ay))))) / xy_sq;
%localLorig = max(abs(gAxgAy ./ AxAy));
%diff = [tfocs_normsq(A_x - A_y) - tfocs_normsq(x - y)];
%if abs(diff) > 1e-12
%  diff
%[xy_sq tfocs_normsq(x - y)]
%end

localL = max(L, localLorig);

%%g_x = apply_linear( g_Ax , 2);
%%localL = 2 * tfocs_dot( x - y, g_x - g_y ) / xy_sq
%size(g_x)t
%size(g_y)
%localL = 2 * tfocs_dot( x - y, g_x - g_y ) / xy_sq;
%g_x = apply_linear(g_Ax, 2);
%localL2 = 2 * tfocs_dot( x - y, g_x - g_y) / xy_sq;


%  g_x = apply_linear(g_Ax, 2);
    q_x = tfocs_dot(xy, g_y + 0.5 * L * xy);% - 0.5 * L * tfocs_normsq(x-y2);
    %localL = min(localL, L + 2 * max( f_x - q_x, 0 ) / xy_sq);
    localL2 = L + 2 * max( (f_x - f_y) - q_x + max([eps(f_x), eps(f_y), eps(q_x), eps(f_x - f_y)]), 0 ) / xy_sq;

if backtrack_simple,
  if localL < localL2
%    [localLorig, localL2, L]
%    [f_x, f_y, q_x]
%    [(f_x - f_y) q_x]
%    [xy_sq]
    global xval
    xval = x;
    global yval;
    global A_x2;
    A_x2 = A_x;
    global A_y2;
    A_y2 = A_y;
    yval = y;
    end
%  [L, localL, localL2]
  localL = min(localL, localL2);
% Should be min, strange stuff
else
[xy_sq];
normlimit = [(abs( xy_sq ) / (backtrack_tol * max(max( abs(tfocs_normsq(x)), abs(tfocs_normsq(y))),1)))];
normlimit2 = [abs( f_y - f_x ) / (backtrack_tol * max(max( abs( f_x ), abs( f_y ) ), 1))];
%theta
[tfocs_normsq(x) tfocs_normsq(y)];
[tfocs_normsq(x) - tfocs_normsq(y)];
%[f_x f_y]
%if normlimit < 0.01 || normlimit2 < 0.01
%localL = Inf;
%xy_sq = 0;
%backtrack_steps = 2;
%theta = 1;
%break
%end
%  localL = max(localL, localL2);
end
%    xd = [f_x - q_x]

%    q_y = f_x + tfocs_dot(-xy, g_x) + 0.5 * L * xy_sq;
%    localL = max(0*localL, L + 2 * max( f_y - q_y, 0 ) / xy_sq);
%    yd = [f_y - q_y]
%
%    xyd = [f_x - f_y]

%    [f_x - q_x f_x - f_y tfocs_dot(xy, g_y)]
%    xy_sq
%    q2_x = -f_y - tfocs_absdot(xy, g_y) - 0.5 * L * xy_sq;
%    localL = max(localL, L + 2 * max( q2_x - -f_x, 0 ) / xy_sq);
%    [(q2_x --f_x) f_x - f_y tfocs_dot(xy, g_y) tfocs_absdot(xy,g_y)]
    %1/L
    %1/localL
    %1/(2 * tfocs_dot( A_x - A_y, g_Ax - g_Ay ) / xy_sq)
%localL3 = max(abs((g_Ax-g_Ax_old)./abs((A_x-A_x_old)+1e-300)));
%localL = max(localL, localL3);
%localL

% Exit if Lipschitz criterion satisfied, or if we hit Lexact
backtrack_steps = backtrack_steps + 1;

if f_x - f_y > 0
  localL = max(L, localL);
end

if localL <= L || localL >= Lexact, do_break=true; return; end
if ~isinf( localL ), 
    %L = min( Lexact, localL );
elseif isinf( localL ), localL = L; end
%L = min( Lexact, max( localL, L / beta ) );
L = min( Lexact, L / beta );

% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.
