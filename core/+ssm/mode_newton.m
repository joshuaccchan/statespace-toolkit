% ssm.mode_newton - the mode of a concave log density by Newton-Raphson, each step
% solved with the banded negative Hessian. See Chan (forthcoming), Section 10.2.2.
%
%   [alphahat, K] = ssm.mode_newton(alpha0, gradK)
%   [alphahat, K, iter] = ssm.mode_newton(alpha0, gradK, 'Tol', tol, 'MaxIter', m)
%
%   alpha0    : n x 1 starting point
%   gradK     : handle; [g, K] = gradK(alpha) returns the gradient g of the log
%               density at alpha and its negative Hessian K, n x n, sparse and banded
%   'Tol'     : stop once max|step| <= Tol (default 1e-4)
%   'MaxIter' : raise if that has not happened after this many steps (default 100)
%   alphahat  : the mode
%   K         : the negative Hessian of the last step, evaluated at the iterate
%               before alphahat
%   iter      : number of steps taken
%
% Each step is alpha + K\g. A NaN step runs to the cap and raises. Written for this
% toolkit.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Sections 10.2.2 and 10.3.3.

function [alphahat, K, iter] = mode_newton(alpha0, gradK, opts)
arguments
    alpha0 (:,1) double
    gradK (1,1) function_handle
    opts.Tol (1,1) double {mustBePositive} = 1e-4
    opts.MaxIter (1,1) double {mustBeInteger, mustBePositive} = 100
end
alphahat = alpha0;
err = Inf;
iter = 0;
while ~(err <= opts.Tol)
    if iter >= opts.MaxIter
        error('ssm:mode_newton:notConverged', ...
            'no convergence to Tol = %g in %d steps (last max|step| = %g)', ...
            opts.Tol, opts.MaxIter, err);
    end
    iter = iter + 1;
    [g, K] = gradK(alphahat);
    alphanew = alphahat + K\g;
    err = max(abs(alphanew - alphahat));
    alphahat = alphanew;
end
end
