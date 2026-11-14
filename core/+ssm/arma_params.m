% ssm.arma_params - posterior draw of the coefficients rho and psi of ARMA(1,1) errors with
% time-varying variances,
%   e_t = rho*e_{t-1} + u_t + psi*u_{t-1},  u_t ~ N(0, 1/w_t),  e_0 = u_0 = 0,
% given e and the precisions w, under the priors rho ~ N(rho0, Vrho) and
% psi ~ N(psi0, Vpsi), each truncated to (-1, 1).
%
%   [rho, psi, u, flag] = ssm.arma_params(e, w, rho, psi, Hyper)
%
%   e     : T x 1 errors
%   w     : T x 1 precisions of u_t, such as exp(-h) or 1./(lam.*exp(h))
%   rho, psi : the current draws; the new draws on output
%   Hyper : struct with rho0, Vrho, psi0 and Vpsi
%   u     : T x 1 errors u_t under the returned rho and psi, filter([1 -rho], [1 psi], e)
%   flag  : 1 if the candidate for psi was accepted
%
% Draws rho, then psi. Stacked over t, H_rho*e = H_psi*u, with H_rho and H_psi lower
% bidiagonal and both of determinant one. Given psi, u_t = z_t - rho*z_{t-1} with
% z = H_psi\e and z_0 = 0, a regression on z_{t-1}: rho is drawn from its normal
% conditional truncated to (-1, 1) by ssm.tnormrnd. Given rho, H_rho*e has MA(1) errors, and
% psi is drawn by ssm.ma_params. Random numbers per call: rand, then those of
% ssm.ma_params.
%
% Trap: when rho is close to -psi, the two lag terms nearly cancel and the posterior has a
% ridge along rho = -psi, along which the chain moves slowly.
%
% Written for this toolkit.
%
% See:
% Zhang, B., Chan, J.C.C. and Cross, J.L. (2020). Stochastic Volatility Models with ARMA
% Innovations: An Application to G7 Inflation Forecasts, International Journal of
% Forecasting, 36(4): 1318-1328.

function [rho, psi, u, flag] = arma_params(e, w, rho, psi, Hyper)
z = filter(1, [1 psi], e);
zl = [0; z(1:end-1)];
Krho = 1/Hyper.Vrho + w'*zl.^2;
rhohat = (Hyper.rho0/Hyper.Vrho + w'*(zl.*z))/Krho;
rho = ssm.tnormrnd(rhohat, 1/Krho, -1, 1);
[psi, u, flag] = ssm.ma_params(e - rho*[0; e(1:end-1)], w, psi, Hyper);
end
