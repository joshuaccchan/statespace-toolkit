function out = uc_sampler(y, m, nsim, burnin, thin)
% uc_sampler - Gibbs sampler for the models of uc_model other than 'ARbound'.
% Blocks: tau (as s = Hpsi\tau when the gap has the MA term) by ssm.simulate_states; a
% constant variance by its inverse-gamma conditional; a random-walk log variance by
% ssm.ksc_rw_h0, then its variance and its start x0, as in the book's UCSV.m; a stationary
% AR(1) log variance by ssm.ksc_ar1_mean, then its variance, phi (independence
% Metropolis-Hastings) and mu, as in Chan (2013)'s UC_MA.m; tau0; and psi by independence
% Metropolis-Hastings at the mode of its conditional. The auxiliary mixture works on
% log(e.^2 + 1e-4), as in the published code.
%   out.phi : nsim x k draws of the free parameters, in the order of m.free
%   out.tau, out.sdh, out.sdg : every thin-th draw of tau, exp(h/2) and exp(g/2)
%   out.last : nsim x 4 draws of [tau_T h_T g_T u_T], the states the forecasts start from
%   out.acc : acceptance rates of the Metropolis-Hastings steps
T = numel(y);
H = ssm.diffmat(T);
% initial values
st.gap = init(m.gap, log(var(y)/2), T);
st.trend = init(m.trend, log(var(y)/20), T);
tau0 = m.tau1.a0; psi = 0;
nsave = floor(nsim/thin);
out.phi = zeros(nsim, numel(m.free));
out.tau = zeros(nsave, T); out.sdh = zeros(nsave, T); out.sdg = zeros(nsave, T);
out.last = zeros(nsim, 4);
acc = struct('psi', 0, 'gap_phi', 0, 'trend_phi', 0);
psic = linspace(-.975, .975, 40)';

for loop = 1:nsim + burnin
    % tau: yt = s + u, and b = G*s - c ~ N(0, diag(v)), with b_1 = tau_1 - tau0
    if m.ma, Hpsi = ssm.diffmat(T, -psi); yt = Hpsi\y; else, Hpsi = speye(T); yt = y; end
    G = H*Hpsi;
    eh = exp(-st.gap.x);
    v = exp(st.trend.x); v(1) = m.tau1.V1a + m.tau1.V1b*v(1);
    iv = 1./v;
    c = [tau0; zeros(T-1,1)];
    K = G'*spdiags(iv, 0, T, T)*G + spdiags(eh, 0, T, T);
    s = ssm.simulate_states(K, G'*(iv.*c) + eh.*yt);
    tau = Hpsi*s;
    u = yt - s;
    b = [tau(1) - tau0; diff(tau)];

    % the gap: its errors u
    [st.gap, a] = draw_logvar(m.gap, st.gap, u, T);
    acc.gap_phi = acc.gap_phi + a;
    % the trend: its innovations b, from t = 2 when tau_1 has its own variance V1a
    if m.tau1.V1a == 0
        [st.trend, a] = draw_logvar(m.trend, st.trend, b/sqrt(m.tau1.V1b), T);
    else
        [st.trend, a] = draw_logvar(m.trend, st.trend, b(2:end), T);
        st.trend.x = [st.trend.x(1); st.trend.x];      % x(1) is not used
    end
    acc.trend_phi = acc.trend_phi + a;

    % tau0 | tau_1
    if m.tau1.b0 > 0
        v1 = m.tau1.V1a + m.tau1.V1b*exp(st.trend.x(1));
        K0 = 1/m.tau1.b0 + 1/v1;
        tau0 = (m.tau1.a0/m.tau1.b0 + tau(1)/v1)/K0 + randn/sqrt(K0);
    end

    % psi | y, tau, h
    if m.ma
        x = y - tau; ehh = exp(-st.gap.x);
        fc = psi_logcond(psic, x, ehh);
        [~, j] = max(fc);
        [pm, Dp] = psi_mode(psic(j), x, ehh);
        pc = pm + sqrt(Dp)*randn;
        ua = rand;
        if abs(pc) < 1
            la = psi_logcond(pc, x, ehh) - psi_logcond(psi, x, ehh) ...
                + .5*(pc - pm)^2/Dp - .5*(psi - pm)^2/Dp;
            if log(ua) < la, psi = pc; acc.psi = acc.psi + 1; end
        end
        u = filter(1, [1 psi], x);                      % the errors under the retained psi
    end

    if loop > burnin
        i = loop - burnin;
        out.phi(i,:) = free_vector(m, st, psi);
        out.last(i,:) = [tau(T) st.gap.x(T) st.trend.x(T) u(T)];
        if mod(i, thin) == 0
            k = i/thin;
            out.tau(k,:) = tau'; out.sdh(k,:) = exp(st.gap.x/2)'; out.sdg(k,:) = exp(st.trend.x/2)';
        end
    end
end
f = fieldnames(acc);
for j = 1:numel(f), acc.(f{j}) = acc.(f{j})/(nsim + burnin); end
out.acc = acc;
end

function s = init(k, lv, T)
s = struct('x', lv*ones(T,1), 's2', .05, 'x0', lv, 'mu', lv, 'phi', .9);
if strcmp(k.type, 'const'), s.s2 = exp(lv); end
end

function [s, accepted] = draw_logvar(k, s, e, T)
% one component: e are its errors (the gap or the trend innovations); s.x is its
% log variance path of length numel(e)
n = numel(e); accepted = 0;
switch k.type
    case 'const'
        s.s2 = 1/gamrnd(k.nu + n/2, 1/(k.S + e'*e/2));
        s.x = log(s.s2)*ones(n,1);
    case 'rw'
        Ystar = log(e.^2 + .0001);
        s.x = ssm.ksc_rw_h0(Ystar, s.x, s.s2, s.x0);
        d = [s.x(1) - s.x0; diff(s.x)];
        s.s2 = 1/gamrnd(k.nu + n/2, 1/(k.S + d'*d/2));
        K0 = 1/k.V0 + 1/s.s2;
        s.x0 = (k.m0/k.V0 + s.x(1)/s.s2)/K0 + randn/sqrt(K0);
    case 'ar1'
        Ystar = log(e.^2 + .0001);
        s.x = ssm.ksc_ar1_mean(Ystar, s.x, s.mu, s.phi, s.s2);
        x = s.x; mu = s.mu; phi = s.phi;
        r = [(x(1) - mu)*sqrt(1 - phi^2); x(2:end) - phi*x(1:end-1) - mu*(1 - phi)];
        s.s2 = 1/gamrnd(k.nu + n/2, 1/(k.S + r'*r/2));
        % phi: the regression of x_t - mu on x_{t-1} - mu as proposal, the
        % stationary density of x_1 in the acceptance ratio (UC_MA.m)
        X = x(1:end-1) - mu; z = x(2:end) - mu;
        D = 1/(1/k.Vphi + X'*X/s.s2);
        phic = D*(k.phi0/k.Vphi + X'*z/s.s2) + sqrt(D)*randn;
        g = @(q) -.5*log(s.s2/(1 - q^2)) - .5*(1 - q^2)/s.s2*(x(1) - mu)^2;
        if abs(phic) < .9999 && exp(g(phic) - g(phi)) > rand
            s.phi = phic; accepted = 1;
        end
        phi = s.phi;
        Dmu = 1/(1/k.V0 + ((n-1)*(1 - phi)^2 + (1 - phi^2))/s.s2);
        muhat = Dmu*(k.m0/k.V0 + (1 - phi^2)/s.s2*x(1) + (1 - phi)/s.s2*sum(x(2:end) - phi*x(1:end-1)));
        s.mu = muhat + sqrt(Dmu)*randn;
end
end

function v = free_vector(m, st, psi)
v = [];
for c = {'gap', 'trend'}
    k = m.(c{1}); s = st.(c{1});
    switch k.type
        case 'const', v = [v log(s.s2)]; %#ok<AGROW>
        case 'rw',    v = [v s.x0 log(s.s2)]; %#ok<AGROW>
        case 'ar1',   v = [v s.mu atanh(s.phi) log(s.s2)]; %#ok<AGROW>
    end
end
if m.ma, v = [v atanh(psi)]; end
end

function f = psi_logcond(q, x, eh)
% log p(psi | y, tau, h) up to a constant, from the MA(1) residuals
% u = filter(1, [1 psi], x) and the N(0,1) prior
f = zeros(numel(q), 1);
for k = 1:numel(q)
    u = filter(1, [1 q(k)], x);
    f(k) = -.5*(eh'*u.^2) - .5*q(k)^2;
end
end

function [pm, Dp] = psi_mode(p, x, eh)
% Newton steps on the log conditional of psi, with the derivatives of the
% residuals v = du/dpsi and w = d2u/dpsi2 from the same recursion
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
        w = filter(1, [1 q], -2*[0; v(1:end-1)]);
        d1 = -sum(eh.*u.*v) - q;
        d2 = -sum(eh.*(v.^2 + u.*w)) - 1;
    end
end
