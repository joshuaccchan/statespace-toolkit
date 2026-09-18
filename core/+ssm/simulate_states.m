% ssm.simulate_states - draws of a state path x ~ N(K\b, inv(K)) given its
% posterior precision K: the precision sampler of Chan and Jeliazkov (2009).
%
%   x = ssm.simulate_states(K, b)
%   [x, xhat] = ssm.simulate_states(K, b, ndraws)
%
%   K      : n x n posterior precision of the stacked path, symmetric positive
%            definite. For a state space model it is banded, so a sparse K makes
%            each draw O(n).
%   b      : n x 1 vector with K*xhat = b. For y = X*x + e, e ~ N(0, Sig), and the
%            prior x ~ N(m, inv(P)) implied by the state equation,
%            K = P + X'*inv(Sig)*X and b = P*m + X'*inv(Sig)*y.
%   ndraws : number of independent draws (default 1)
%   x      : n x ndraws matrix of draws
%   xhat   : n x 1 posterior mean K\b
%
% Written for this toolkit; it generalizes the inline draw of the published code.
% The draw is xhat + chol(K,'lower')'\randn(n, ndraws), with xhat = K\b computed
% separately: the spelling of that code, so that a sampler calling this function
% reproduces the published one draw for draw (tests/unit/test_simulate_states.m).

function [x, xhat] = simulate_states(K, b, ndraws)
if nargin < 3
    ndraws = 1;
end
xhat = K\b;
x = xhat + chol(K,'lower')'\randn(size(K,1), ndraws);
end
