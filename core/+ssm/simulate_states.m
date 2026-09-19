% ssm.simulate_states - draws from N(mu, inv(K)) given the precision matrix K: the
% precision sampler of Chan and Jeliazkov (2009). See Algorithm 9.1 of Chan
% (forthcoming) for a textbook treatment.
%
%   alpha = ssm.simulate_states(mu, K)
%   alpha = ssm.simulate_states(mu, K, ndraws)
%
%   mu     : n x 1 mean
%   K      : n x n symmetric positive definite precision matrix. For the states of
%            a linear Gaussian state space model (Theorem 9.1 of the book),
%            K = P + Z'*inv(R)*Z and mu = K\(P*b + Z'*inv(R)*y), with the prior
%            precision P = G'*inv(Q)*G and prior mean b. K is banded, so a sparse K
%            makes each draw O(n).
%   ndraws : number of independent draws (default 1)
%   alpha  : n x ndraws matrix of draws
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.1.1 and Algorithm 9.1.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.

function alpha = simulate_states(mu, K, ndraws)
if nargin < 3
    ndraws = 1;
end
alpha = mu + chol(K,'lower')'\randn(size(K,1), ndraws);
end
