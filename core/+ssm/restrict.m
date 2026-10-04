% ssm.restrict - imposes the linear restriction M*x = z on draws x from N(xhat, K^{-1}):
% each column becomes x + U*((M*U)\(z - M*x)), with U = K^{-1}*M', a draw from the
% distribution of x given M*x = z (Algorithm 2.6 of Rue and Held, 2005; Algorithm 2 of
% Chan, Poon and Zhu, 2023). Applied to the mean xhat, it gives the conditional mean.
% Written for this toolkit.
%
%   x = ssm.restrict(x, C, M, z)
%   [x, U, MU] = ssm.restrict(x, C, M, z)
%
%   x  : n x ndraws draws from N(xhat, K^{-1}), or the n x 1 mean xhat
%   C  : lower Cholesky factor of K, the third output of ssm.simulate_states
%   M  : r x n restriction matrix of rank r, sparse or full
%   z  : r x 1
%   U  : n x r matrix K^{-1}*M'
%   MU : r x r matrix M*K^{-1}*M'; the conditional covariance is K^{-1} - U*(MU\U')
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
% Algorithm 2.
% Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications,
% Chapman & Hall/CRC, Algorithm 2.6.

function [x, U, MU] = restrict(x, C, M, z)
U = C'\(C\full(M'));
MU = M*U;
x = x + U*(MU\(z - M*x));
end
