function test_simulate_states
% (1) Draw for draw: each anchor script in tests/fixtures/, run whole on a short
%     chain, stores the same draws when its inline draw of the states, given the
%     posterior mean the script computes, is replaced by a call to
%     ssm.simulate_states. The anchors are sp_code's UC.m (the published code of
%     Chan and Jeliazkov 2009) and chan-jeliazkov-2009's UC.m, linreg_tvp.m and
%     DFM.m. Each patch must match exactly one line.
% (2) Moments: on a simulated local level model, K\b and inv(K) equal the Kalman
%     smoother's moments, and 20,000 draws match its means, variances and lag-one
%     covariances.
% (3) Several draws at once equal the same number of single draws.
root = getappdata(0, 'ssm_repo_root');
fx = fullfile(root, 'tests', 'fixtures');
vis = get(0, 'DefaultFigureVisible');
set(0, 'DefaultFigureVisible', 'off');
restore = onCleanup(@() set(0, 'DefaultFigureVisible', vis));

% --- (1) the four anchors ------------------------------------------------------
sp = fullfile(fx, 'bvar-toolkit', 'replications', 'chan_jeliazkov2009_statespace', 'legacy', 'sp_code');
cj = fullfile(fx, 'chan-jeliazkov-2009');
A = struct('dir', {}, 'files', {}, 'script', {}, 'seed', {}, 'run', {}, 'swap', {}, 'out', {});
A(1) = struct('dir', sp, 'files', {{'UC.m', 'SVRW.m', 'USCPI.csv'}}, 'script', 'UC.m', 'seed', 1, ...
    'run', {{'clear; clc;', ''; 'nloop = 11000;', 'nloop = 60;'; 'burnin = 1000;', 'burnin = 10;'}}, ...
    'swap', {{'Ctau = chol(Ktau,''lower'');', ''; ...
              'tau = tauhat + Ctau''\randn(T,1);', '    tau = ssm.simulate_states(tauhat, Ktau);'}}, ...
    'out', {{'store_tau', 'store_h', 'store_omega2tau', 'store_omega2h'}});
A(2) = struct('dir', cj, 'files', {{'UC.m', 'USCPI.csv'}}, 'script', 'UC.m', 'seed', [], ...
    'run', {{'clear; clc; rng(42);', 'rng(42);'; 'nsim = 20000;', 'nsim = 50;'; 'burnin = 1000;', 'burnin = 10;'}}, ...
    'swap', {{'Ctau = chol(Ktau,''lower'');', ''; ...
              'tau = tau_hat + Ctau''\randn(T,1);', '    tau = ssm.simulate_states(tau_hat, Ktau);'}}, ...
    'out', {{'store_tau', 'store_theta'}});
A(3) = struct('dir', cj, 'files', {{'linreg_tvp.m', 'SURform.m', 'USPCE_OutputGap.csv'}}, 'script', 'linreg_tvp.m', 'seed', [], ...
    'run', {{'clear; clc; rng(42);', 'rng(42);'; 'nsim = 20000; burnin = 1000;', 'nsim = 50; burnin = 10;'}}, ...
    'swap', {{'CKbeta = chol(Kbeta, ''lower'');', ''; ...
              'beta = beta_hat + (CKbeta'')\randn(k*T,1);', '    beta = ssm.simulate_states(beta_hat, Kbeta);'}}, ...
    'out', {{'store_beta', 'store_theta'}});
A(4) = struct('dir', cj, 'files', {{'DFM.m', 'FRED-MD.csv'}}, 'script', 'DFM.m', 'seed', [], ...
    'run', {{'clear; clc; rng(42);', 'rng(42);'; 'nsim = 20000;', 'nsim = 40;'; 'burnin = 1000;', 'burnin = 5;'}}, ...
    'swap', {{'f = f_hat + chol(Kf,''lower'')'' \ randn(T*r,1);', '    f = ssm.simulate_states(f_hat, Kf);'}}, ...
    'out', {{'store_F', 'store_A', 'store_sig2', 'store_omega2', 'store_phi'}});

for ii = 1:numel(A)
    a = A(ii);
    leg = run_anchor(a, a.run, sprintf('legacy%d', ii));
    new = run_anchor(a, [a.run; a.swap], sprintf('ssm%d', ii));
    for v = a.out
        assert(isequal(leg.(v{1}), new.(v{1})), ...
            'simulate_states: %s of %s differs from the inline draw', v{1}, fullfile(a.dir, a.script));
    end
end

% --- (2) moments against the Kalman smoother -----------------------------------
rng(11, 'twister');
T = 50; sig2 = 0.8; omega2 = 0.15; tau0 = 1.2;
tau = tau0 + cumsum(sqrt(omega2)*randn(T,1));
y = tau + sqrt(sig2)*randn(T,1);
H = ssm.diffmat(T); HH = H'*H;
K = HH/omega2 + speye(T)/sig2;                 % tau_1 ~ N(tau0, omega2), as in the UC anchors
b = tau0/omega2*(HH*ones(T,1)) + y/sig2;
[m_ks, P_ks, C_ks] = kalman_local_level(y, sig2, omega2, tau0);

S = inv(full(K));
assert(max(abs(diag(S) - P_ks)) < 1e-10 && max(abs(diag(S,1) - C_ks)) < 1e-10, ...
    'simulate_states: inv(K) differs from the smoother covariances');
tauhat = K\b;
assert(max(abs(tauhat - m_ks)) < 1e-10, 'simulate_states: K\\b differs from the smoother mean');
nd = 20000;
rng(12, 'twister');
X = ssm.simulate_states(tauhat, K, nd);
assert(isequal(size(X), [T nd]), 'simulate_states: wrong size of draws');
zm = (mean(X,2) - m_ks)./sqrt(P_ks/nd);
Xc = X - mean(X,2);
zv = (sum(Xc.^2,2)/(nd-1) - P_ks)./(P_ks*sqrt(2/nd));
c1 = sum(Xc(1:end-1,:).*Xc(2:end,:), 2)/(nd-1);
zc = (c1 - C_ks)./sqrt((P_ks(1:end-1).*P_ks(2:end) + C_ks.^2)/nd);
assert(max(abs([zm; zv; zc])) < 4.5, ...
    'simulate_states: draws miss the smoother moments (max |z| = %.2f)', max(abs([zm; zv; zc])));

% --- (3) several draws at once --------------------------------------------------
rng(5, 'twister'); Xm = ssm.simulate_states(tauhat, K, 4);
rng(5, 'twister'); X1 = zeros(T, 4);
for j = 1:4
    X1(:,j) = ssm.simulate_states(tauhat, K);
end
assert(isequal(Xm, X1), 'simulate_states: ndraws = 4 differs from four single draws');
end

function [ms, Ps, Cs] = kalman_local_level(y, sig2, omega2, tau0)
% Kalman filter and RTS smoother for y_t = tau_t + e_t, tau_t = tau_{t-1} + u_t,
% tau_1 ~ N(tau0, omega2). Returns smoothed means, variances, and lag-one
% covariances Cov(tau_t, tau_{t+1} | y).
T = numel(y);
a = zeros(T,1); P = zeros(T,1); ap = zeros(T,1); Pp = zeros(T,1);
for t = 1:T
    if t == 1
        ap(t) = tau0; Pp(t) = omega2;
    else
        ap(t) = a(t-1); Pp(t) = P(t-1) + omega2;
    end
    g = Pp(t)/(Pp(t) + sig2);
    a(t) = ap(t) + g*(y(t) - ap(t));
    P(t) = (1 - g)*Pp(t);
end
ms = a; Ps = P; Cs = zeros(T-1,1);
for t = T-1:-1:1
    J = P(t)/Pp(t+1);
    ms(t) = a(t) + J*(ms(t+1) - ap(t+1));
    Ps(t) = P(t) + J^2*(Ps(t+1) - Pp(t+1));
    Cs(t) = J*Ps(t+1);
end
end
