function test_ksc_ar1_mean
% (1) Equal draw for draw under one seed to bvar.sv.ksc_ar1_mean, held verbatim in
%     tests/fixtures/. (2) Equal to SV.m of chan2013_joe_masv, the draw of Chan
%     (2013), with the arguments reordered: the same mixture indicators, the same
%     state of the random number stream, and the path to rounding, since SV.m forms
%     the prior mean in another order of operations.
root = getappdata(0, 'ssm_repo_root');

rng(41, 'twister');                            % fixed test data
T = 137;
e = randn(T,1).*exp(0.3*randn(T,1));
ystar = log(e.^2 + 1e-4);
h_in = 0.2*randn(T,1);
mu = 0.4; rho = 0.93; sig2 = 0.05;

rng(7, 'twister');
[h, S] = ssm.ksc_ar1_mean(ystar, h_in, mu, rho, sig2);
s1 = rng;
assert(isequal(size(h), [T 1]) && all(isfinite(h)) && all(ismember(S, 1:7)), 'ksc_ar1_mean: bad draw');

fx = fullfile(root, 'tests', 'fixtures', 'bvar-toolkit', 'core');
addpath(fx); c = onCleanup(@() rmpath(fx));
rng(7, 'twister');
[h_b, S_b] = bvar.sv.ksc_ar1_mean(ystar, h_in, mu, rho, sig2);
assert(isequal(h, h_b) && isequal(S, S_b), 'ksc_ar1_mean: differs from bvar.sv.ksc_ar1_mean');
clear c

leg = fullfile(root, 'replications', 'chan2013_joe_masv', 'legacy', 'MASV_matlab');
addpath(leg); c2 = onCleanup(@() rmpath(leg));
rng(7, 'twister');
[h_l, S_l] = SV(ystar, h_in, rho, sig2, mu);
s2 = rng;
assert(isequal(S, S_l) && isequal(s1, s2), 'ksc_ar1_mean: indicators or stream differ from SV.m');
assert(max(abs(h - h_l)) <= 1e-10*max(1, max(abs(h_l))), 'ksc_ar1_mean: path differs from SV.m');
end
