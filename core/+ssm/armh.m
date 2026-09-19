% ssm.armh - one accept-reject Metropolis-Hastings update of a state vector, with
% a Gaussian proposal at the mode of the target: the algorithm of Chan (2017).
%
%   [alpha, accept] = ssm.armh(alpha, logf, gradK)
%   [alpha, accept] = ssm.armh(alpha, logf, gradK, 'c_reject', 3, 'ForceAccept', false)
%
%   alpha         : n x 1 current state; on output the new state
%   logf          : handle; logf(alpha) is the log target density, up to a constant
%   gradK         : handle; [g, K] = gradK(alpha) returns the gradient of logf and
%                   its negative Hessian, sparse and banded (see ssm.mode_newton)
%   'c_reject'    : envelope constant c = c_reject*f(alphahat)/g(alphahat)
%                   (default 3)
%   'ForceAccept' : take the candidate whatever the MH ratio (default false), for
%                   the first sweeps of a chain; the uniform is drawn regardless
%   'Start'       : starting point of the mode search (default: the current state)
%   'Tol', 'MaxIterMode' : passed to ssm.mode_newton (defaults 1e-4 and 100). A Tol
%                   above 1e-3 is refused: the search starts from the current
%                   state, and a loose tolerance lets the proposal depend on it.
%   'MaxIterAR'   : raise after this many rejected candidates (default 1000)
%   accept        : true if the MH step took the candidate
%
% The proposal is g = N(alphahat, inv(K)) at the mode alphahat. Candidates from g
% are kept with probability min{f/(c g), 1}, and the survivor passes an MH step.
% For a textbook discussion, see Chan (forthcoming), Sections 6.2.3, 10.2.2 and
% 10.3.3.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Sections 6.2.3, 10.2.2 and 10.3.3.
% Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with Time-Varying
% Parameters: An Application to Inflation Modeling, Journal of Business and
% Economic Statistics, 35(1): 17-28.

function [alpha, accept] = armh(alpha, logf, gradK, opts)
arguments
    alpha (:,1) double
    logf (1,1) function_handle
    gradK (1,1) function_handle
    opts.c_reject (1,1) double {mustBePositive, mustBeFinite} = 3
    opts.ForceAccept (1,1) logical = false
    opts.Start double = []
    opts.Tol (1,1) double {mustBePositive} = 1e-4
    opts.MaxIterMode (1,1) double {mustBeInteger, mustBePositive} = 100
    opts.MaxIterAR (1,1) double {mustBeInteger, mustBePositive} = 1000
end
if opts.Tol > 1e-3
    error('ssm:armh:looseTol', ['Tol = %g is looser than 1e-3. The mode search ' ...
        'starts from the current state, so a loose tolerance lets the proposal ' ...
        'depend on it.'], opts.Tol);
end
if isempty(opts.Start)
    opts.Start = alpha;
end
opts.Start = opts.Start(:);
[ahat, K] = ssm.mode_newton(opts.Start, gradK, 'Tol', opts.Tol, 'MaxIter', opts.MaxIterMode);
C = chol(K,'lower');
logc = log(opts.c_reject) + logf(ahat);        % log c; the kernel of g is 1 at ahat
n = length(alpha);

% accept-reject step: screen candidates from g
accepted = false;
iter = 0;
while ~accepted
    iter = iter + 1;
    if iter > opts.MaxIterAR
        error('ssm:armh:arNotAccepted', ...
            'no candidate accepted in %d draws at c_reject = %g', opts.MaxIterAR, opts.c_reject);
    end
    ac = ahat + C'\randn(n,1);
    dc = ac - ahat;
    lac = logf(ac) + .5*dc'*K*dc - logc;       % log f/(c g) at the candidate
    accepted = lac > log(rand);
end

% MH step
d = alpha - ahat;
la = logf(alpha) + .5*d'*K*d - logc;           % log f/(c g) at the current state
if la < 0
    lalpha = 0;
elseif lac < 0
    lalpha = -la;
else
    lalpha = lac - la;
end
accept = lalpha > log(rand) || opts.ForceAccept;
if accept
    alpha = ac;
end
end
