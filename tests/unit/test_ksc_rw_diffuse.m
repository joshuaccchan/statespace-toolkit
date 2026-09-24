function test_ksc_rw_diffuse
% (1) Equal draw for draw under one seed to bvar.sv.ksc_rw_diffuse, held verbatim in
%     tests/fixtures/. (2) Equal to SVRW.m of chan_clark_koop2018_jmcb_trendie with
%     h0 = 0: the same state of the random number stream, and the path to rounding,
%     since SVRW.m solves for the mean with backslash, which factors the precision a
%     second time.
root = getappdata(0, 'ssm_repo_root');

rng(43, 'twister');                            % fixed test data
T = 131;
e = randn(T,1).*exp(0.4*randn(T,1));
ystar = log(e.^2 + 1e-4);
h_in = 0.3*randn(T,1);
omega2h = 0.08; Vh = 100;

rng(9, 'twister');
[h, S] = ssm.ksc_rw_diffuse(ystar, h_in, omega2h, Vh);
s1 = rng;
assert(isequal(size(h), [T 1]) && all(isfinite(h)) && all(ismember(S, 1:7)), 'ksc_rw_diffuse: bad draw');

fx = fullfile(root, 'tests', 'fixtures', 'bvar-toolkit', 'core');
addpath(fx); c = onCleanup(@() rmpath(fx));
rng(9, 'twister');
[h_b, S_b] = bvar.sv.ksc_rw_diffuse(ystar, h_in, omega2h, Vh);
assert(isequal(h, h_b) && isequal(S, S_b), 'ksc_rw_diffuse: differs from bvar.sv.ksc_rw_diffuse');
clear c

leg = fullfile(root, 'replications', 'chan_clark_koop2018_jmcb_trendie', 'legacy');
addpath(leg); c2 = onCleanup(@() rmpath(leg));
rng(9, 'twister');
h_l = SVRW(ystar, h_in, omega2h, 0, Vh);
s2 = rng;
assert(isequal(s1, s2), 'ksc_rw_diffuse: random number stream differs from SVRW.m');
assert(max(abs(h - h_l)) <= 1e-10*max(1, max(abs(h_l))), 'ksc_rw_diffuse: path differs from SVRW.m');
end
