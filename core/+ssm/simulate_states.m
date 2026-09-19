% ssm.simulate_states - draws from N(alphahat, K^{-1}), where K is the precision matrix
% and alphahat solves K*alphahat = c: Algorithm 1 of Chan and Jeliazkov (2009), which
% obtains the mean and the draws from one Cholesky factor of K. It is a more efficient
% implementation of Algorithm 9.1 of Chan (forthcoming), which takes the mean as an
% input and so needs a separate solve with K to compute it.
%
%   [alpha, alphahat] = ssm.simulate_states(K, c)
%   [alpha, alphahat] = ssm.simulate_states(K, c, ndraws)
%
%   K        : n x n symmetric positive definite precision matrix. For the states of
%              a linear Gaussian state space model (Theorem 9.1 of the book),
%              K = P + Z'*inv(R)*Z and c = P*b + Z'*inv(R)*y, with the prior
%              precision P = G'*inv(Q)*G and prior mean b. K is banded, so a sparse K
%              makes each draw O(n).
%   c        : n x 1 vector
%   ndraws   : number of independent draws (default 1)
%   alpha    : n x ndraws matrix of draws
%   alphahat : n x 1 mean, K\c
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.1.1 and Algorithm 9.1.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120, Algorithm 1.

function [alpha, alphahat] = simulate_states(K, c, ndraws)
if nargin < 3
    ndraws = 1;
end
C = chol(K,'lower');
alphahat = C'\(C\c);
alpha = alphahat + C'\randn(size(K,1), ndraws);
end
