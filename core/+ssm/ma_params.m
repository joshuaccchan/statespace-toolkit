% ssm.ma_params - posterior draw of the coefficient psi of MA(1) errors with time-varying
% variances,
%   x_t = u_t + psi*u_{t-1},  u_t ~ N(0, 1/w_t),  u_0 = 0,
% given x and the precisions w, under the prior psi ~ N(psi0, Vpsi) truncated to (-1, 1).
%
%   [psi, u, flag] = ssm.ma_params(x, w, psi, Hyper)
%
%   x     : T x 1 errors
%   w     : T x 1 precisions of u_t, such as exp(-h) or 1./(lam.*exp(h))
%   psi   : the current draw; the new draw on output
%   Hyper : struct with psi0 and Vpsi
%   u     : T x 1 errors u_t under the returned psi, filter(1, [1 psi], x)
%   flag  : 1 if the candidate was accepted
%
% Stacked over t, x = H_psi*u with H_psi lower bidiagonal, and H_psi has determinant one, so
% the conditional density of psi is exact. The draw is an independence Metropolis-Hastings
% step whose proposal is N(psihat, Dpsi): psihat is the mode of the conditional density and
% Dpsi is minus the inverse of its second derivative there. Newton steps find the mode,
% starting from the best of 40 points on [-0.975, 0.975], so the search does not depend on
% the current draw. Random numbers per call: randn, then rand.
%
% Trap: the conditional density has thicker tails than the normal proposal, so a chain
% started far from the mode can keep its starting value for many sweeps. From psi = 0, with
% the mode at 0.62 and a conditional standard deviation of 0.057, no candidate is accepted
% in 20,000 draws. Start psi near its mode.
%
% Written for this toolkit. ssm.models.uc and ssm.arma_params call it.
%
% See:
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models with Application to
% Inflation Forecast, Journal of Econometrics, 176(2): 162-172.

function [psi, u, flag] = ma_params(x, w, psi, Hyper)
psi0 = Hyper.psi0; Vpsi = Hyper.Vpsi;
psig = linspace(-.975, .975, 40)';
fg = logcond(psig, x, w, psi0, Vpsi);
[~, j] = max(fg);
[pm, Dp] = psi_mode(psig(j), x, w, psi0, Vpsi);
pc = pm + sqrt(Dp)*randn;
ua = rand;
flag = 0;
if abs(pc) < 1
    la = logcond(pc, x, w, psi0, Vpsi) - logcond(psi, x, w, psi0, Vpsi) ...
        + .5*(pc - pm)^2/Dp - .5*(psi - pm)^2/Dp;
    if log(ua) < la, psi = pc; flag = 1; end
end
u = filter(1, [1 psi], x);
end

function f = logcond(q, x, w, psi0, Vpsi)
% log p(psi | x, w) up to a constant, at each element of q
f = zeros(numel(q), 1);
for k = 1:numel(q)
    u = filter(1, [1 q(k)], x);
    f(k) = -.5*(w'*u.^2) - .5*(q(k) - psi0)^2/Vpsi;
end
end

function [pm, Dp] = psi_mode(p, x, w, psi0, Vpsi)
% Newton steps on the log conditional of psi, with the derivatives v = du/dpsi and
% w2 = d2u/dpsi2 from the same recursion as u
for it = 1:100
    [d1, d2] = derivs(p);
    if d2 >= 0, break; end
    pn = min(max(p - d1/d2, -.999), .999);
    if abs(pn - p) < 1e-10, p = pn; break; end
    p = pn;
end
pm = p;
[~, d2] = derivs(p);
if d2 < 0, Dp = -1/d2; else, Dp = .05^2; end
    function [d1, d2] = derivs(q)
        u = filter(1, [1 q], x);
        v = filter(1, [1 q], -[0; u(1:end-1)]);
        w2 = filter(1, [1 q], -2*[0; v(1:end-1)]);
        d1 = -sum(w.*u.*v) - (q - psi0)/Vpsi;
        d2 = -sum(w.*(v.^2 + u.*w2)) - 1/Vpsi;
    end
end
