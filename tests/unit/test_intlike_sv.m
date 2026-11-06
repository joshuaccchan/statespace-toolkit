function test_intlike_sv
% ssm.intlike_sv against intlike_sv.m of chan_grant2016_eneco_garchsv, copied to a
% temporary folder.
% (1) With the legacy tolerance 'Tol', .01, at two parameter settings, each from its own
%     Newton starting point: the log likelihood and the state of the random number
%     stream are bitwise equal.
% (2) Each invalid input raises its error, a far start with 'MaxIter', 1 raises
%     notConverged, and a start at which the first step overflows raises nonFinite.
% (3) The default and the legacy tolerances agree within Monte Carlo error, and four
%     far-apart starts agree more closely at the default tolerance than at the legacy one.
% (4) The second output nse lies within 25 percent of the standard deviation of 100
%     repeated estimates.
% (5) T = 3: the mean of 20 estimates lies within 4 standard errors of the exact log
%     likelihood, with the log-volatilities integrated out by integral3.
root = getappdata(0, 'ssm_repo_root');
leg = fullfile(root, 'replications', 'chan_grant2016_eneco_garchsv', 'legacy');
d = tempname; mkdir(d);
cleanUp = onCleanup(@() remove_dir(d));
copyfile(fullfile(leg, 'intlike_sv.m'), d);
addpath(d);

% --- (1) bitwise equal to the legacy function at its tolerance ------------------------
rng(41, 'twister');                            % fixed test data
T = 137; mu = 0.1;
h = zeros(T,1); h(1) = 0.5 + sqrt(0.05/(1 - 0.95^2))*randn;
for t = 2:T, h(t) = 0.5 + 0.95*(h(t-1) - 0.5) + sqrt(0.05)*randn; end
y = mu + exp(h/2).*randn(T,1);
s2 = (y - mu).^2;
pars = {0.5, 0.95, 0.05, log(var(y))*ones(T,1); -0.3, 0.8, 0.2, h};
for k = 1:size(pars,1)
    [muh, phih, omegah2, ht] = pars{k,:};
    rng(7, 'twister');
    ll_l = intlike_sv(s2, muh, phih, omegah2, ht, 500);
    sL = rng;
    rng(7, 'twister');
    [ll, nse] = ssm.intlike_sv(s2, muh, phih, omegah2, ht, 500, 'Tol', .01);
    sN = rng;
    assert(isequal(ll, ll_l), 'intlike_sv: setting %d differs from intlike_sv.m', k);
    assert(isequal(sL, sN), 'intlike_sv: setting %d leaves the random number stream in another state', k);
    assert(isreal(ll) && isfinite(ll) && isfinite(nse) && nse > 0, 'intlike_sv: setting %d gives a bad value', k);
end
rmpath(d);

% --- (2) the errors ------------------------------------------------------------------
f = @(varargin) ssm.intlike_sv(varargin{:});
good = {s2, 0.5, 0.95, 0.05, h, 50};
bad = {1, s2', 'badS2'; 1, [s2(1:end-1); NaN], 'badS2'; 1, [-1; s2(2:end)], 'badS2'; 1, zeros(0,1), 'badS2'; ...
       2, NaN, 'badMuh'; 2, [0 1], 'badMuh'; ...
       3, 1, 'badPhih'; 3, 1.01, 'badPhih'; 3, NaN, 'badPhih'; ...
       4, 0, 'badOmegah2'; 4, -1, 'badOmegah2'; 4, Inf, 'badOmegah2'; ...
       5, h(1:end-1), 'badHt'; 5, h', 'badHt'; 5, [h(1:end-1); NaN], 'badHt'; ...
       6, 0, 'badM'; 6, 2.5, 'badM'; 6, [10 20], 'badM'};
for k = 1:size(bad,1)
    a = good; a{bad{k,1}} = bad{k,2};
    expect_error(@() f(a{:}), ['ssm:intlike_sv:' bad{k,3}]);
end
msg = expect_error(@() f(s2, 0.5, 0.95, 0.05, 20*ones(T,1), 50, 'MaxIter', 1), 'ssm:intlike_sv:notConverged');
assert(contains(msg, 'last max|step|'), 'intlike_sv: the notConverged message gives no step size');
expect_error(@() f(s2, 0.5, 0.95, 0.05, -800*ones(T,1), 50), 'ssm:intlike_sv:nonFinite');

% --- (3) the tolerance and the starting point ----------------------------------------
rng(7, 'twister'); [la, na] = ssm.intlike_sv(s2, 0.5, 0.95, 0.05, h, 500);
rng(7, 'twister'); lb = ssm.intlike_sv(s2, 0.5, 0.95, 0.05, h, 500, 'Tol', .01);
assert(abs(la - lb) <= 3*na, 'intlike_sv: default and legacy tolerances differ by %.2e, nse %.2e', la - lb, na);
starts = [-20 0 20 50]; v = zeros(numel(starts), 2);
for s = 1:numel(starts)
    rng(7, 'twister'); v(s,1) = ssm.intlike_sv(s2, 0.5, 0.95, 0.05, starts(s)*ones(T,1), 500);
    rng(7, 'twister'); v(s,2) = ssm.intlike_sv(s2, 0.5, 0.95, 0.05, starts(s)*ones(T,1), 500, 'Tol', .01);
end
assert(range(v(:,1)) < range(v(:,2)) && range(v(:,1)) < 1e-5, ...
    'intlike_sv: far starts spread %.2e at the default tolerance, %.2e at the legacy one', range(v(:,1)), range(v(:,2)));

% --- (4) the standard error against the spread of repeated estimates -----------------
rng(12, 'twister');
R = 100; ll = zeros(R,1); ns = ll;
for r = 1:R
    [ll(r), ns(r)] = ssm.intlike_sv(s2, 0.5, 0.95, 0.05, h, 500);
end
ratio = median(ns)/std(ll);
assert(ratio > 0.75 && ratio < 1.25, 'intlike_sv: median nse is %.2f times the spread of %d estimates', ratio, R);

% --- (5) T = 3 against the exact log likelihood ----------------------------------------
y3 = [0.8; -1.5; 0.3]; muh = 0.2; phih = 0.9; omegah2 = 0.1;
V = omegah2/(1 - phih^2)*phih.^abs((1:3)' - (1:3));   % stationary covariance of h
iV = inv(V); cV = 1/sqrt((2*pi)^3*det(V));
lik = @(h, yt) exp(-.5*log(2*pi) - .5*h - .5*yt^2*exp(-h));
f3 = @(h1, h2, h3) lik(h1, y3(1)).*lik(h2, y3(2)).*lik(h3, y3(3)).*cV.*exp(-.5*( ...
    iV(1,1)*(h1-muh).^2 + iV(2,2)*(h2-muh).^2 + iV(3,3)*(h3-muh).^2 ...
    + 2*iV(1,2)*(h1-muh).*(h2-muh) + 2*iV(1,3)*(h1-muh).*(h3-muh) + 2*iV(2,3)*(h2-muh).*(h3-muh)));
b = muh + 8*sqrt(V(1,1))*[-1 1];               % 8 standard deviations of h_t on either side
lexact = log(integral3(f3, b(1), b(2), b(1), b(2), b(1), b(2), 'AbsTol', 0, 'RelTol', 1e-8));
rng(11, 'twister');
R = 20; est = zeros(R,1);
for r = 1:R
    est(r) = ssm.intlike_sv(y3.^2, muh, phih, omegah2, muh*ones(3,1), 20000);
end
se = std(est)/sqrt(R);
assert(abs(mean(est) - lexact) <= 4*se, ...
    'intlike_sv: mean of %d estimates %.6f, exact %.6f, standard error %.1e', R, mean(est), lexact, se);
end

function msg = expect_error(fn, id)
msg = '';
try
    fn();
catch err
    assert(strcmp(err.identifier, id), 'expected %s, got %s: %s', id, err.identifier, err.message);
    msg = err.message;
    return
end
error('expected the error %s, and none was raised', id);
end

function remove_dir(d)
if any(strcmp(strsplit(path, pathsep), d)), rmpath(d); end
rmdir(d, 's');
end
