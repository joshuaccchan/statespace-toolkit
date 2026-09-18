function test_ksc_rw_h0
% A finite draw of the right size, equal draw for draw under one seed to
% bvar.sv.ksc_rw_h0, held verbatim in tests/fixtures/. bvar-toolkit tests that
% function against five published SVRW.m copies; the SVRW.m files archived here
% are different samplers (tests/variant_map.md).
root = getappdata(0, 'ssm_repo_root');

rng(41, 'twister');                            % fixed test data
T = 137;
e = randn(T,1).*exp(0.3*randn(T,1));
ystar = log(e.^2 + 1e-4);
h_in = 0.2*randn(T,1);
sig2 = 0.12;                                   % state innovation VARIANCE
h0 = 0.4;

rng(7, 'twister');
h = ssm.ksc_rw_h0(ystar, h_in, sig2, h0);
assert(isequal(size(h), [T 1]) && all(isfinite(h)), 'ksc_rw_h0: bad draw');

fx = fullfile(root, 'tests', 'fixtures', 'bvar-toolkit', 'core');
addpath(fx); c = onCleanup(@() rmpath(fx));    % bvar.sv.ksc_rw_h0 from the fixture
rng(7, 'twister');
h_bvar = bvar.sv.ksc_rw_h0(ystar, h_in, sig2, h0);
assert(isequal(h, h_bvar), 'ksc_rw_h0: differs from bvar.sv.ksc_rw_h0 under the same seed');
end
