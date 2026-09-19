function test_armh
% (1) The book's sample_SVM_h_ARMH.m, a fixture: 40 chained sweeps on simulated
%     SV-in-mean data give bitwise the same draws and acceptances as ssm.armh with
%     the book's target, gradient and negative Hessian, and leave the random
%     number stream in the same state.
% (2) The published sample_CSV.m of Chan (2023), a fixture, and (5) the inline
%     step of UC_SVM.m in chan2017_jbes_svm, run whole on a short chain. These
%     write the Newton step as K\(K*h + g) where ssm.mode_newton takes h + K\g, so
%     the modes, and with them the draws, agree to rounding; the acceptances and
%     the random number stream agree exactly. Measured on 18 September 2026: the
%     draws differ by at most 8.9e-16 over 50 sweeps of sample_CSV and 3.5e-13
%     over 150 sweeps of UC_SVM.m, which accepted 114 log-volatility candidates.
% (3) Geweke joint-distribution test on the CSV model: the kernel leaves the joint
%     distribution of (h, s2) unchanged at c_reject = 0.2, 1, 3 and 20 and under a
%     forced accept at a valid envelope, and it detects a wrong conditional and a
%     forced accept below the envelope bound.
% (4) A loose Tol, the two caps and bad c_reject values raise, and ntry counts the
%     accept-reject candidates: a positive integer, larger on average at a larger
%     c_reject.
% (5) runs last: the published script samples (mu, phi) in a loop with no cap,
%     which a broken kernel can send into an endless search, so (1)-(4) must
%     catch a broken kernel first.
root = getappdata(0, 'ssm_repo_root');
fx = fullfile(root, 'tests', 'fixtures');
tol_csv = 1e-12;                                % measured 8.9e-16
tol_svm = 1e-10;                                % measured 3.5e-13

% --- (1) the book's sample_SVM_h_ARMH.m, bitwise -------------------------------
book = fullfile(fx, 'bayesian-macroeconometrics', 'code', 'matlab', 'chapter10');
rng(31, 'twister');
T = 150; alp = .15; mu = .3; sigh2 = .03; h0 = -.5;
htrue = h0 + cumsum(sqrt(sigh2)*randn(T,1));
y = mu + alp*exp(htrue) + exp(htrue/2).*randn(T,1);
H = ssm.diffmat(T); HH = H'*H;
nsw = 40; hstart = htrue;           % a far start sticks: the book's step has no forced accept
addpath(book); c = onCleanup(@() rmpath(book));
rng(32, 'twister'); h = hstart; Hl = zeros(T,nsw); al = zeros(nsw,1);
for i = 1:nsw
    [h, al(i)] = sample_SVM_h_ARMH(y, alp, mu, h, h0, sigh2, HH); Hl(:,i) = h;
end
sl = rng;
clear c
logf = @(x) -0.5*(x - h0)'*HH*(x - h0)/sigh2 -0.5*sum(x) ...
    -0.5*sum(exp(-x).*(y - mu - alp.*exp(x)).^2);
gradK = @(x) book_gradK(x, y, alp, mu, h0, sigh2, HH);
rng(32, 'twister'); h = hstart; Hs = zeros(T,nsw); as = zeros(nsw,1);
for i = 1:nsw
    [h, as(i)] = ssm.armh(h, logf, gradK); Hs(:,i) = h;
end
ss = rng;
assert(isequal(Hs, Hl) && isequal(as, al), 'armh: differs from the book''s sample_SVM_h_ARMH');
assert(isequal(ss.State, sl.State), 'armh: random number stream differs from the book''s');
assert(any(as), 'armh: the book comparison never accepted a candidate');

% --- (2) the published sample_CSV.m, to rounding --------------------------------
csv = fullfile(fx, 'bvar-toolkit', 'replications', 'chan2023_joe_mlvarsv', 'legacy', 'utility');
rng(1, 'twister');
T = 120; n = 15; rho = .95; sigh2 = .05;
htrue = zeros(T,1); htrue(1) = sqrt(sigh2/(1-rho^2))*randn;
for t = 2:T, htrue(t) = rho*htrue(t-1) + sqrt(sigh2)*randn; end
s2 = exp(htrue).*chi2rnd(n,T,1);
Hrho = speye(T) - rho*sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HiSH = Hrho'*sparse(1:T,1:T,[(1-rho^2)/sigh2; 1/sigh2*ones(T-1,1)])*Hrho;
logf = @(x) -.5*x'*HiSH*x - n/2*sum(x) - .5*exp(-x)'*s2;
gradK = @(x) deal(-n/2 + .5*(s2./exp(x)) - HiSH*x, HiSH + sparse(1:T,1:T,.5*(s2./exp(x))));
nsw = 50;
addpath(csv); c = onCleanup(@() rmpath(csv));
rng(42, 'twister'); h = log(s2/n); Hl = zeros(T,nsw); al = zeros(nsw,1);
for i = 1:nsw
    [h, al(i)] = sample_CSV(s2, rho, sigh2, h, n, i == 1); Hl(:,i) = h;
end
sl = rng;
clear c
rng(42, 'twister'); h = log(s2/n); Hs = zeros(T,nsw); as = zeros(nsw,1);
for i = 1:nsw
    [h, as(i)] = ssm.armh(h, logf, gradK, 'Tol', 1e-3, 'MaxIterMode', 1e4, 'ForceAccept', i == 1);
    Hs(:,i) = h;
end
ss = rng;
d_csv = max(abs(Hs(:) - Hl(:)));
assert(isequal(as, al) && isequal(ss.State, sl.State), ...
    'armh: acceptances or random number stream differ from sample_CSV');
assert(d_csv < tol_csv, 'armh: draws differ from sample_CSV by %.3g', d_csv);

% --- (3) Geweke joint-distribution test ------------------------------------------
Tg = 40; ng = 5; rg = .95; sg = .04; M = 4000; burn = 200; nb = 40;
for c_reject = [0.2 1 3 20]
    z = geweke_z(Tg, ng, ng, rg, sg, M, burn, nb, c_reject, false);
    assert(max(abs(z)) < 4, 'armh: joint distribution moved at c_reject = %g (max|z| = %.2f)', ...
        c_reject, max(abs(z)));
end
z = geweke_z(Tg, ng, ng+1, rg, sg, M, burn, nb, 3, false);
assert(max(abs(z)) > 20, 'armh: a wrong conditional went undetected (max|z| = %.2f)', max(abs(z)));
z = geweke_z(Tg, ng, ng, rg, sg, M, burn, nb, 3, true);
assert(max(abs(z)) < 4, 'armh: forced accept at a valid envelope moved the joint (max|z| = %.2f)', ...
    max(abs(z)));
z = geweke_z(Tg, ng, ng, rg, sg, M, burn, nb, 0.2, true);
assert(max(abs(z)) > 8, 'armh: forced accept below the envelope bound went undetected (max|z| = %.2f)', ...
    max(abs(z)));

% --- (4) bad options raise --------------------------------------------------------
h = log(s2/n);
named = {{'Tol', 1e-2}, 'ssm:armh:looseTol'; ...
         {'MaxIterMode', 1}, 'ssm:mode_newton:notConverged'; ...
         {'c_reject', 1e12, 'MaxIterAR', 5}, 'ssm:armh:arNotAccepted'};
for ii = 1:size(named, 1)
    try
        ssm.armh(h, logf, gradK, named{ii,1}{:});
        error('test:noThrow', 'armh: %s did not raise', named{ii,2});
    catch err
        assert(strcmp(err.identifier, named{ii,2}), 'armh: expected %s, got %s', ...
            named{ii,2}, err.identifier);
    end
end
for c_reject = {0, -3, Inf, NaN, [1 10]}
    try
        ssm.armh(h, logf, gradK, 'c_reject', c_reject{1});
        error('test:noThrow', 'armh: c_reject = %s did not raise', mat2str(c_reject{1}));
    catch err
        assert(~strcmp(err.identifier, 'test:noThrow'), '%s', err.message);
    end
end
rng(7, 'twister'); nt = zeros(100, 2);
for i = 1:100
    [~, ~, nt(i,1)] = ssm.armh(h, logf, gradK, 'c_reject', 1);
    [~, ~, nt(i,2)] = ssm.armh(h, logf, gradK, 'c_reject', 20);
end
assert(all(nt(:) >= 1 & nt(:) == round(nt(:))), 'armh: ntry must be a positive integer');
assert(mean(nt(:,2)) > mean(nt(:,1)), ...
    'armh: fewer candidates on average at c_reject = 20 (%.2f) than at 1 (%.2f)', ...
    mean(nt(:,2)), mean(nt(:,1)));

% --- (5) UC_SVM.m of Chan (2017), run whole, to rounding ------------------------
a = struct('dir', fullfile(root, 'replications', 'chan2017_jbes_svm', 'legacy'), ...
    'files', {{'UC_SVM.m', 'SURform.m', 'proplam.m', 'USCPI.csv'}}, 'script', 'UC_SVM.m', ...
    'seed', 1, 'out', {{'store_theta', 'store_tau', 'store_alp', 'store_h', 'counth', 'countlam'}});
runfix = {'clear; clc;', '', ''; 'nloop = 55000;', 'nloop = 150;', ''; 'burnin = 5000;', 'burnin = 10;', ''};
swap = {'errh = 1; ht = h;', { ...
    '    logf_h = @(x) -.5*(x-deltah)''*HinvSH*(x-deltah) -.5*sum(x) + ...', ...
    '        -.5*exp(-x)''*(y-tau-alp.*exp(x)).^2;', ...
    '    gradK_h = @(x) deal(-.5 + .5*(s2./exp(x)) - .5*(alp.^2.*exp(x)) - HinvSH*(x-deltah), ...', ...
    '        HinvSH + spdiags(.5*(s2./exp(x)) + .5*(alp.^2.*exp(x)),0,T,T));', ...
    '    [h, acc_h] = ssm.armh(h, logf_h, gradK_h, ''Tol'', 1e-3, ''MaxIterMode'', 1e4, ''ForceAccept'', loop == 1);', ...
    '    if acc_h', '        exph = exp(h);', '        counth = counth + 1;', '    end', ''}, ...
    '%% sample beta'};
leg = run_anchor(a, runfix, 'legacy');
new = run_anchor(a, [runfix; swap], 'ssm');
d_svm = 0;
for v = {'store_theta', 'store_tau', 'store_alp', 'store_h'}
    d_svm = max(d_svm, max(abs(leg.(v{1})(:) - new.(v{1})(:))));
end
assert(leg.counth == new.counth && leg.countlam == new.countlam ...
    && isequal(leg.rngstate.State, new.rngstate.State), ...
    'armh: acceptances or random number stream differ from UC_SVM.m');
assert(d_svm < tol_svm, 'armh: UC_SVM.m draws differ by %.3g', d_svm);
end

function [grad, Kh] = book_gradK(ht, y, alp, mu, h0, sigh2, HH)
% the gradient and negative Hessian exactly as sample_SVM_h_ARMH.m computes them
T = size(ht, 1);
s2 = (y - mu).^2;
exp_ht = exp(ht);
curv1 = 0.5 * alp.^2 .* exp_ht;
curv2 = 0.5 * s2 ./ exp_ht;
g = - 0.5 - curv1 + curv2;
grad = g - (HH * (ht - h0))/sigh2;
G  = sparse(1:T, 1:T, -curv1 - curv2, T, T);
Kh = HH / sigh2 - G;
end

function z = geweke_z(T, n_gen, n_ker, rho, sigh2, M, burn, nb, c_reject, forced)
% z-statistics comparing four functionals of h between the marginal-conditional
% simulator (h from the prior, s2 given h) and the successive-conditional one (s2
% given h, then h through ssm.armh); batch means give the chain's variance
rng(20260910, 'twister');
A = zeros(M,4);
for i = 1:M, A(i,:) = geweke_funcs(ar1_prior(T, rho, sigh2)); end
Hrho = speye(T) - rho*sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HiSH = Hrho'*sparse(1:T,1:T,[(1-rho^2)/sigh2; 1/sigh2*ones(T-1,1)])*Hrho;
rng(20260910, 'twister');
h = ar1_prior(T, rho, sigh2);
B = zeros(M,4);
for i = 1:(M+burn)
    s2 = exp(h).*chi2rnd(n_gen,T,1);
    logf = @(x) -.5*x'*HiSH*x - n_ker/2*sum(x) - .5*exp(-x)'*s2;
    gradK = @(x) deal(-n_ker/2 + .5*(s2./exp(x)) - HiSH*x, HiSH + sparse(1:T,1:T,.5*(s2./exp(x))));
    h = ssm.armh(h, logf, gradK, 'c_reject', c_reject, 'ForceAccept', forced);
    if i > burn, B(i-burn,:) = geweke_funcs(h); end
end
mA = mean(A); vA = var(A)/M;
K = floor(M/nb);
bm = squeeze(mean(reshape(B(1:K*nb,:), nb, K, []), 1));
z = (mA - mean(bm))./sqrt(vA + var(bm)/K);
end

function h = ar1_prior(T, rho, sigh2)
h = zeros(T,1);
h(1) = sqrt(sigh2/(1-rho^2))*randn;
for t = 2:T, h(t) = rho*h(t-1) + sqrt(sigh2)*randn; end
end

function f = geweke_funcs(h)
f = [mean(h), h(1), h(end), mean(h.^2)];
end
