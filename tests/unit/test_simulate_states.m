function test_simulate_states
% (1) Draw for draw: each anchor script in tests/fixtures/, run whole on a short
%     chain, stores the same draws when its inline state draw is replaced by a
%     call to ssm.simulate_states. The anchors are sp_code's UC.m (the published
%     code of Chan and Jeliazkov 2009) and chan-jeliazkov-2009's UC.m,
%     linreg_tvp.m and DFM.m. Each patch must match exactly one line.
% (2) Moments: on a simulated local level model, K\b equals the Kalman smoother
%     mean, and 20,000 draws match its variances and lag-one covariances.
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
    'swap', {{'Ctau = chol(Ktau,''lower'');', ''; 'tauhat = Ktau\(invSigy*y);', ''; ...
              'tau = tauhat + Ctau''\randn(T,1);', '    tau = ssm.simulate_states(Ktau, invSigy*y);'}}, ...
    'out', {{'store_tau', 'store_h', 'store_omega2tau', 'store_omega2h'}});
A(2) = struct('dir', cj, 'files', {{'UC.m', 'USCPI.csv'}}, 'script', 'UC.m', 'seed', [], ...
    'run', {{'clear; clc; rng(42);', 'rng(42);'; 'nsim = 20000;', 'nsim = 50;'; 'burnin = 1000;', 'burnin = 10;'}}, ...
    'swap', {{'tau_hat = Ktau\(tau0/omega2*HHiota + y/sig2);', ''; 'Ctau = chol(Ktau,''lower'');', ''; ...
              'tau = tau_hat + Ctau''\randn(T,1);', '    tau = ssm.simulate_states(Ktau, tau0/omega2*HHiota + y/sig2);'}}, ...
    'out', {{'store_tau', 'store_theta'}});
A(3) = struct('dir', cj, 'files', {{'linreg_tvp.m', 'SURform.m', 'USPCE_OutputGap.csv'}}, 'script', 'linreg_tvp.m', 'seed', [], ...
    'run', {{'clear; clc; rng(42);', 'rng(42);'; 'nsim = 20000; burnin = 1000;', 'nsim = 50; burnin = 10;'}}, ...
    'swap', {{'CKbeta = chol(Kbeta, ''lower'');', ''; 'beta_hat = Kbeta\(P*repmat(beta0,T,1) + Zy/sig2);', ''; ...
              'beta = beta_hat + (CKbeta'')\randn(k*T,1);', '    beta = ssm.simulate_states(Kbeta, P*repmat(beta0,T,1) + Zy/sig2);'}}, ...
    'out', {{'store_beta', 'store_theta'}});
A(4) = struct('dir', cj, 'files', {{'DFM.m', 'FRED-MD.csv'}}, 'script', 'DFM.m', 'seed', [], ...
    'run', {{'clear; clc; rng(42);', 'rng(42);'; 'nsim = 20000;', 'nsim = 40;'; 'burnin = 1000;', 'burnin = 5;'}}, ...
    'swap', {{'f_hat = Kf \ (kron(speye(T), A'' * iSig) * y);', ''; ...
              'f = f_hat + chol(Kf,''lower'')'' \ randn(T*r,1);', '    f = ssm.simulate_states(Kf, kron(speye(T), A'' * iSig) * y);'}}, ...
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
nd = 20000;
rng(12, 'twister');
[X, xhat] = ssm.simulate_states(K, b, nd);
assert(max(abs(xhat - m_ks)) < 1e-10, 'simulate_states: K\\b differs from the smoother mean');
assert(isequal(size(X), [T nd]), 'simulate_states: wrong size of draws');
zm = (mean(X,2) - m_ks)./sqrt(P_ks/nd);
Xc = X - mean(X,2);
zv = (sum(Xc.^2,2)/(nd-1) - P_ks)./(P_ks*sqrt(2/nd));
c1 = sum(Xc(1:end-1,:).*Xc(2:end,:), 2)/(nd-1);
zc = (c1 - C_ks)./sqrt((P_ks(1:end-1).*P_ks(2:end) + C_ks.^2)/nd);
assert(max(abs([zm; zv; zc])) < 4.5, ...
    'simulate_states: draws miss the smoother moments (max |z| = %.2f)', max(abs([zm; zv; zc])));

% --- (3) several draws at once --------------------------------------------------
rng(5, 'twister'); Xm = ssm.simulate_states(K, b, 4);
rng(5, 'twister'); X1 = zeros(T, 4);
for j = 1:4
    X1(:,j) = ssm.simulate_states(K, b);
end
assert(isequal(Xm, X1), 'simulate_states: ndraws = 4 differs from four single draws');
end

function ssmtest_out = run_anchor(ssmtest_a, ssmtest_patches, ssmtest_tag)
% Copy the anchor's files to a fresh folder, patch the script line by line, run
% it whole, and return the named variables. Variable names are prefixed so that
% the script, which runs in this workspace, cannot overwrite them.
ssmtest_dir = tempname; mkdir(ssmtest_dir);
ssmtest_clean = onCleanup(@() rmdir(ssmtest_dir, 's'));
for ssmtest_f = ssmtest_a.files
    copyfile(fullfile(ssmtest_a.dir, ssmtest_f{1}), ssmtest_dir);
end
ssmtest_txt = patch_lines(fileread(fullfile(ssmtest_a.dir, ssmtest_a.script)), ...
    ssmtest_patches, ssmtest_a.script);
ssmtest_file = fullfile(ssmtest_dir, ['anchor_' erase(ssmtest_a.script, '.m') '_' ssmtest_tag '.m']);
ssmtest_fid = fopen(ssmtest_file, 'w');
fwrite(ssmtest_fid, ssmtest_txt);
fclose(ssmtest_fid);
if ~isempty(ssmtest_a.seed)
    rng(ssmtest_a.seed, 'twister');
end
evalc('run(ssmtest_file)');
close all force
ssmtest_out = struct();
for ssmtest_v = ssmtest_a.out
    ssmtest_out.(ssmtest_v{1}) = eval(ssmtest_v{1});
end
end

function txt = patch_lines(txt, patches, script)
% Replace whole lines, each identified by the statement it starts with; the file
% keeps its own line endings.
if contains(txt, sprintf('\r\n')), eol = sprintf('\r\n'); else, eol = newline; end
lines = strsplit(txt, eol, 'CollapseDelimiters', false);
for p = 1:size(patches, 1)
    hit = find(startsWith(strtrim(lines), patches{p,1}));
    assert(isscalar(hit), 'simulate_states: patch "%s" matches %d lines of %s', ...
        patches{p,1}, numel(hit), script);
    lines{hit} = patches{p,2};
end
txt = strjoin(lines, eol);
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
