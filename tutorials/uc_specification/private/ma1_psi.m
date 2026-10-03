function [psi, accepted] = ma1_psi(psi, x, iv, Vpsi, bound)
% ma1_psi - one independence Metropolis-Hastings step for the coefficient of an
% MA(1) error, x_t = u_t + psi*u_{t-1}, u_t ~ N(0, 1/iv_t), u_0 = 0, with prior
% psi ~ N(0, Vpsi) on (-bound, bound). The proposal is normal at the mode of the
% conditional density, found by Newton steps from the best of 40 grid points, with
% the curvature there as its precision; the derivatives of u come from the same
% recursion as u.
%   Random numbers: randn, then rand.
grid = linspace(-.975, .975, 40)'*bound;
fc = arrayfun(@(q) logcond(q, x, iv, Vpsi), grid);
[~, j] = max(fc);
[pm, Dp] = mode_newton(grid(j), x, iv, Vpsi, bound);
pc = pm + sqrt(Dp)*randn;
ua = rand;
accepted = false;
if abs(pc) < bound
    la = logcond(pc, x, iv, Vpsi) - logcond(psi, x, iv, Vpsi) ...
        + .5*(pc - pm)^2/Dp - .5*(psi - pm)^2/Dp;
    if log(ua) < la, psi = pc; accepted = true; end
end
end

function f = logcond(q, x, iv, Vpsi)
u = filter(1, [1 q], x);
f = -.5*(iv'*u.^2) - .5*q^2/Vpsi;
end

function [p, Dp] = mode_newton(p, x, iv, Vpsi, bound)
for it = 1:100
    [d1, d2] = derivs(p);
    if d2 >= 0, break; end
    pn = min(max(p - d1/d2, -bound + 1e-3), bound - 1e-3);
    if abs(pn - p) < 1e-10, p = pn; break; end
    p = pn;
end
[~, d2] = derivs(p);
if d2 < 0, Dp = -1/d2; else, Dp = .05^2; end
    function [d1, d2] = derivs(q)
        u = filter(1, [1 q], x);
        v = filter(1, [1 q], -[0; u(1:end-1)]);
        w = filter(1, [1 q], -2*[0; v(1:end-1)]);
        d1 = -sum(iv.*u.*v) - q/Vpsi;
        d2 = -sum(iv.*(v.^2 + u.*w)) - 1/Vpsi;
    end
end
