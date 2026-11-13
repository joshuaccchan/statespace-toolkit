% ssm.sv_params - posterior draw of the parameters (mu, phi, sig2) of stationary AR(1)
% log-volatilities with mean mu,
%   h_t = mu + phi*(h_{t-1} - mu) + u_t,  u_t ~ N(0, sig2),  h_1 ~ N(mu, sig2/(1-phi^2)),
% given their paths, under the priors sig2 ~ IG(nuh, Sh), phi ~ N(phi0, Vphi) truncated to
% (-phi_bnd, phi_bnd) and mu ~ N(mu0, Vmu).
%
%   [mu, phi, sig2, flag_phi] = ssm.sv_params(h, mu, phi, Hyper)
%   [mu, phi, sig2, flag_phi] = ssm.sv_params(h, mu, phi, Hyper, phi_bnd)
%
%   h        : T x (n+r) log-volatility paths; the first n columns have mean mu
%              (n = numel(mu)), the last r have mean zero and get no mean draw
%   mu       : n x 1 current means; the new draw on output
%   phi      : (n+r) x 1 current AR(1) coefficients; the new draw on output
%   Hyper    : struct with nuh, Sh, phi0, Vphi, mu0 and Vmu
%   phi_bnd  : bound on |phi|; default .999, and 1 gives the prior support (-1, 1)
%   sig2     : (n+r) x 1 innovation variances
%   flag_phi : (n+r) x 1, 1 where the phi candidate was accepted
%
% Draws sig2, then phi, then mu. The phi candidate comes from N(phi_hat, 1/Kphi), the
% conditional without the density of h_1. It is rejected unless |phi| < phi_bnd, and is
% otherwise accepted with the Metropolis-Hastings probability that adds that density.
%
% Trap: the mu block runs only when every element of mu is nonzero (`if mu~=0`); a single
% exact zero skips the draw of all of mu.
%
% Code-identical to bvar.sv.sv_params in bvar-toolkit, and the two must stay so.
%
% See:
% Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
% VARs with Stochastic Volatility, Journal of Business and Economic
% Statistics, 42(2): 825-837.

function [mu,phi,sig2,flag_phi] = sv_params(h,mu,phi,Hyper,phi_bnd)
if nargin < 5
    phi_bnd = .999;     % OISV canonical truncation bound
end
n = size(mu,1);
[T,npr] = size(h);
r = npr - n;

    % sample sig2
tmp_h = [h(:,1:n)-repmat(mu',T,1), h(:,n+1:end)];
e_h = [tmp_h(1,:).*sqrt(1-phi.^2)'; tmp_h(2:end,:)-repmat(phi',T-1,1).*tmp_h(1:end-1,:)];
sig2 = 1./gamrnd(Hyper.nuh+T/2,1./(Hyper.Sh + sum(e_h.^2)'/2));

    % sample phi
Kphi = 1./Hyper.Vphi + sum(tmp_h(1:T-1,:).^2)'./sig2;
phi_hat = (Hyper.phi0./Hyper.Vphi + sum(tmp_h(1:T-1,:).*tmp_h(2:T,:))'./sig2)./Kphi;
phic = phi_hat + 1./sqrt(Kphi).*randn(n+r,1);
flag_phi = zeros(n+r,1);
for ii = 1:n+r
    g_phi = @(x) .5*log(1-x^2) -.5*(1-x^2)/sig2(ii)*tmp_h(1,ii)^2;
    if abs(phic(ii))<phi_bnd
        alpMH = exp(g_phi(phic(ii))-g_phi(phi(ii)));
        if alpMH>rand
            phi(ii) = phic(ii);
            flag_phi(ii) = 1;
        end
    end
end

    % sample mu
if mu~=0
Kmu = 1./Hyper.Vmu + ((1-phi(1:n).^2) + (T-1)*(1-phi(1:n)).^2)./sig2(1:n);
mu_hat = (Hyper.mu0./Hyper.Vmu + (1-phi(1:n).^2)./sig2(1:n).*h(1,1:n)' ...
   +(1-phi(1:n))./sig2(1:n).*sum(h(2:end,1:n)-repmat(phi(1:n)',T-1,1).*h(1:end-1,1:n))')./Kmu;
mu = mu_hat + 1./sqrt(Kmu).*randn(n,1);
end
end
