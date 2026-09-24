function test_ksc_rw_noncentered
% ssm.ksc_rw_noncentered against SVRW_gam.m of chan2018_er_spectest, in a copy whose
% two solves use the Cholesky factor of the draw, as the library's do.
% (1) One call: every output and the state of the random number stream are bitwise
%     equal.
% (2) UCSV_gam.m run whole for 400 sweeps on the quarterly CPI inflation of
%     examples/data, once with the copy of SVRW_gam.m and once with both of its
%     calls replaced by ssm.ksc_rw_noncentered: every stored draw, every log
%     ordinate and the stream are bitwise equal. The clock seed of UCSV_gam.m is
%     patched out, because once rand('state',...) has run, rng(seed) is refused.
root = getappdata(0, 'ssm_repo_root');
leg = fullfile(root, 'replications', 'chan2018_er_spectest', 'legacy');
d = tempname; mkdir(d);
cleanUp = onCleanup(@() remove_dir(d));
copyfile(fullfile(leg, 'SVRW_gam.m'), d);
copyfile(fullfile(leg, 'UCSV_gam.m'), d);
one_factor(fullfile(d, 'SVRW_gam.m'));

% --- (1) one call --------------------------------------------------------------
rng(41, 'twister');                            % fixed test data
T = 137;
e = randn(T,1).*exp(0.3*randn(T,1));
Ystar = log(e.^2 + 1e-4);
htilde = randn(T,1); h0 = 0.3; omegah = -0.4;
b0 = 0.1; Vh0 = 10; Vh = 10; Vomegah = 0.2;
addpath(d);
rng(7, 'twister');
L = cell(1,5); [L{:}] = SVRW_gam(Ystar, htilde, h0, omegah, b0, Vh0, Vh, Vomegah);
sL = rng;
rng(7, 'twister');
N = cell(1,5); [N{:}] = ssm.ksc_rw_noncentered(Ystar, htilde, h0, omegah, b0, Vh0, Vh, Vomegah);
sN = rng;
names = {'htilde', 'h0', 'omegah', 'omegahhat', 'Domegah'};
for k = 1:5
    assert(isequal(L{k}, N{k}), 'ksc_rw_noncentered: %s differs from SVRW_gam.m', names{k});
end
assert(isequal(sL, sN), 'ksc_rw_noncentered: the random number stream ends in another state');
assert(isequal(size(N{1}), [T 1]) && all(isfinite(N{1})) && N{5} > 0, ...
    'ksc_rw_noncentered: bad draw');
rmpath(d);

% --- (2) UCSV_gam.m run whole -----------------------------------------------------
csv = fullfile(root, 'examples', 'data', 'USCPI_quarterly.csv');
a = struct('dir', d, 'files', {{'UCSV_gam.m', 'SVRW_gam.m'}}, 'script', 'UCSV_gam.m', ...
    'seed', 3, 'out', {{'store_theta', 'store_tau', 'store_h', 'store_g', 'store_lpden', 'lBF'}});
setup = {'valh = 0;', {sprintf('y = readmatrix(''%s'', ''Range'', ''C2:C312'');', csv), ...
                       'nloop = 400; burnin = 100;', 'valh = 0;'}; ...
         'rand(''state''', ''};
swap = {'[htilde h0 omegah omegahhat Domegah] = SVRW_gam(', ...
            '    [htilde,h0,omegah,omegahhat,Domegah] = ssm.ksc_rw_noncentered(Ystar,htilde,h0,omegah,b0,Vh0,Vh,Vomegah);'; ...
        '[gtilde g0 omegag omegaghat Domegag] = SVRW_gam(', ...
            '    [gtilde,g0,omegag,omegaghat,Domegag] = ssm.ksc_rw_noncentered(Ystar,gtilde,g0,omegag,c0,Vg0,Vg,Vomegag);'};
Lr = run_anchor(a, setup, 'legacy');
a.files = {'UCSV_gam.m'};                      % the copy of SVRW_gam.m is not there to call
Nr = run_anchor(a, [setup; swap], 'ssm');
for v = a.out
    assert(isequal(Lr.(v{1}), Nr.(v{1})), ...
        'ksc_rw_noncentered: %s of UCSV_gam.m differs from the run with SVRW_gam.m', v{1});
end
assert(isequal(Lr.rngstate, Nr.rngstate), ...
    'ksc_rw_noncentered: UCSV_gam.m leaves the random number stream in another state');
assert(all(isfinite(Nr.lBF)), 'ksc_rw_noncentered: non-finite log Bayes factor');
end

function remove_dir(d)
if any(strcmp(strsplit(path, pathsep), d)), rmpath(d); end
rmdir(d, 's');
end

function one_factor(file)
% The two solves of SVRW_gam.m, by the Cholesky factor that its draws use. Each
% string must occur exactly once; the draw lines factor the same matrix again and
% obtain the same factor.
subs = {'htildehat = Kh\(invOmega*omegah*(Ystar-dconst-h0));', ...
        'CKh = chol(Kh,''lower''); htildehat = (CKh'')\(CKh\(invOmega*omegah*(Ystar-dconst-h0)));'; ...
        'betahat = invDbeta\(invVbeta*[b0;0] + XbetainvOmega*(Ystar-dconst));', ...
        'CDbeta = chol(invDbeta,''lower''); betahat = (CDbeta'')\(CDbeta\(invVbeta*[b0;0] + XbetainvOmega*(Ystar-dconst)));'};
fid = fopen(file, 'r');
txt = char(fread(fid, Inf, '*uint8')');
fclose(fid);
for k = 1:size(subs,1)
    assert(numel(strfind(txt, subs{k,1})) == 1, 'one_factor: expected exactly one %s', subs{k,1});
    txt = strrep(txt, subs{k,1}, subs{k,2});
end
fid = fopen(file, 'w');
fwrite(fid, uint8(txt));
fclose(fid);
end
