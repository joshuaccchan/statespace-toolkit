function test_sv
% ssm.models.sv must reproduce, draw for draw, the samplers of guides/sv_gibbs (SV) and
% guides/sv_t (SV-t), whose loops are copied below; draw a random-walk log-volatility with
% h_1 ~ N(h0, Vh), as a direct draw with that prior, by shifting ssm.ksc_rw_diffuse by h0;
% recover the coefficients of MA(1) and ARMA(1,1) errors on generated data; run every
% combination of its options, storing h_T, e_T and u_T of each draw; and reject malformed
% input.

offset = 1e-4; nsim = 300; burnin = 50;
rng(42, 'twister');
T = 300; h = zeros(T,1); h(1) = sqrt(.04/(1 - .97^2))*randn;
for t = 2:T, h(t) = .97*h(t-1) + sqrt(.04)*randn; end
y = .1 + exp(h/2).*randn(T,1);
yt = .1 + sqrt(1./gamrnd(2.5, 2/5, T, 1)).*exp(h/2).*randn(T,1);

% the loop of guides/sv_gibbs, section 1
mu0 = 0; Vmu = 10;
hyp = struct('nuh', 5, 'Sh', .16, 'phi0', .97, 'Vphi', .1^2, 'mu0', 1, 'Vmu', 10);
rng(7, 'twister');
muh = log(var(y)); phi = .98; sig2h = .04; hh = muh*ones(T,1);
ref = zeros(nsim, 4);
for isim = 1:burnin + nsim
    Kmu = 1/Vmu + sum(exp(-hh));
    mu = (mu0/Vmu + exp(-hh)'*y)/Kmu + randn/sqrt(Kmu);
    ystar = log((y - mu).^2 + offset);
    hh = ssm.ksc_ar1_mean(ystar, hh, muh, phi, sig2h);
    [muh, phi, sig2h] = ssm.sv_params(hh, muh, phi, hyp, 1);
    if isim > burnin, ref(isim - burnin, :) = [mu muh phi sig2h]; end
end
r = ssm.models.sv(y, 'nsim', nsim, 'burnin', burnin, 'seed', 7);
assert(isequal(r.params, ref), 'sv: the draws differ from the loop of guides/sv_gibbs');
assert(isequal(r.param_names, {'mu', 'mu_h', 'phi', 'sig2_h'}), 'sv: the parameter names differ, SV');

% the loop of guides/sv_t, section 1
rng(8, 'twister');
muh = log(var(yt)); phi = .98; sig2h = .04; nu = 5; hh = muh*ones(T,1); lam = ones(T,1);
ref = zeros(nsim, 5);
for isim = 1:burnin + nsim
    w = 1./(lam.*exp(hh));
    Kmu = 1/Vmu + sum(w);
    mu = (mu0/Vmu + w'*yt)/Kmu + randn/sqrt(Kmu);
    ystar = log((yt - mu).^2./lam + offset);
    hh = ssm.ksc_ar1_mean(ystar, hh, muh, phi, sig2h);
    e2 = (yt - mu).^2.*exp(-hh);
    lam = 1./gamrnd((nu + 1)/2, 2./(nu + e2));
    nu = ssm.nu_studentt(lam, nu, 100);
    [muh, phi, sig2h] = ssm.sv_params(hh, muh, phi, hyp, 1);
    if isim > burnin, ref(isim - burnin, :) = [mu muh phi sig2h nu]; end
end
r = ssm.models.sv(yt, 'dist', 't', 'nsim', nsim, 'burnin', burnin, 'seed', 8);
assert(isequal(r.params, ref), 'sv: the draws differ from the loop of guides/sv_t');

% the random walk shifted by h0 against the direct draw with h_1 ~ N(h0, Vh): the same
% mixture indicators and normal draws, and the precision sampler written out
rng(30, 'twister'); n = 120; h0 = 1.3; Vh = 2; s2 = .05;
ystar = log((exp(h0/2 + cumsum(sqrt(s2)*randn(n,1))/2).*randn(n,1)).^2 + offset);
hc = h0 + .1*randn(n,1);
rng(31, 'twister');
[g, S] = ssm.ksc_rw_diffuse(ystar - h0, hc - h0, s2, Vh);
rng(31, 'twister'); rand(n,1); z = randn(n,1);
mui = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;
sigma2i = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
H = ssm.diffmat(n);
iO = spdiags([1/Vh; ones(n-1,1)/s2], 0, n, n); iS = spdiags(1./sigma2i(S)', 0, n, n);
K = H'*iO*H + iS;
hd = K\(iS*(ystar - mui(S)') + H'*iO*[h0; zeros(n-1,1)]) + chol(K, 'lower')'\z;
assert(max(abs(h0 + g - hd)) < 1e-12, 'sv: the shifted random walk differs from the draw with h_1 ~ N(h0, Vh)');

% the sweep of the header with a random-walk log-volatility, away from the default prior
pr = struct('h0', 1.5, 'Vh', 3, 'nuh', 4, 'Sh', .03);
rng(9, 'twister');
hh = log(var(y))*ones(T,1); sig2h = .01;
ref = zeros(nsim, 2);
for isim = 1:burnin + nsim
    w = exp(-hh);
    Kmu = 1/Vmu + sum(w);
    mu = (mu0/Vmu + w'*y)/Kmu + randn/sqrt(Kmu);
    ystar = log((y - mu).^2 + offset);
    hh = pr.h0 + ssm.ksc_rw_diffuse(ystar - pr.h0, hh - pr.h0, sig2h, pr.Vh);
    dh = diff(hh);
    sig2h = 1/gamrnd(pr.nuh + (T-1)/2, 1/(pr.Sh + dh'*dh/2));
    if isim > burnin, ref(isim - burnin, :) = [mu sig2h]; end
end
r = ssm.models.sv(y, 'vol', 'rw', 'prior', pr, 'nsim', nsim, 'burnin', burnin, 'seed', 9);
assert(isequal(r.params, ref), 'sv: the draws differ from the sweep of the header, random walk');

% MA(1) and ARMA(1,1) errors on generated data: the 99 percent intervals contain mu and the
% coefficients
truth = {'ma', [0 .5]; 'arma', [.6 .3]};
for c = 1:2
    rng(40 + c, 'twister'); n = 600;
    hg = zeros(n,1); hg(1) = sqrt(.04/(1 - .97^2))*randn;
    for t = 2:n, hg(t) = .97*hg(t-1) + sqrt(.04)*randn; end
    yg = 2 + filter([1 truth{c,2}(2)], [1 -truth{c,2}(1)], exp(hg/2).*randn(n,1));
    r = ssm.models.sv(yg, 'errors', truth{c,1}, 'nsim', 2000, 'burnin', 500, 'seed', 1);
    j = find(ismember(r.param_names, {'mu', 'rho', 'psi'}));
    q = quantile(r.params(:,j), [.005; .995], 1);
    tv = [2 truth{c,2}(end - numel(j) + 2:end)];
    assert(all(q(1,:) < tv & tv < q(2,:)), 'sv: a 99 percent interval misses the truth, %s', truth{c,1});
end

% every combination of the options runs, with the parameters and outputs it documents
for vol = {'ar1', 'rw'}
    for errors = {'iid', 'ma', 'arma'}
        for dist = {'normal', 't'}
            for has_mean = [true false]
                r = ssm.models.sv(y, 'vol', vol{1}, 'errors', errors{1}, 'dist', dist{1}, 'mean', has_mean, ...
                    'nsim', 20, 'burnin', 5, 'seed', 2, 'thin', 4);
                names = {'mu', 'rho', 'psi', 'mu_h', 'phi', 'sig2_h', 'nu'};
                used = [has_mean, strcmp(errors{1}, 'arma'), ~strcmp(errors{1}, 'iid'), ...
                    strcmp(vol{1}, 'ar1'), strcmp(vol{1}, 'ar1'), true, strcmp(dist{1}, 't')];
                assert(isequal(r.param_names, names(used)) && all(isfinite(r.params(:))) ...
                    && isequal(size(r.draws.h), [5 T]) && isfield(r, 'lam_mean') == strcmp(dist{1}, 't'), ...
                    'sv: the output differs from the documentation, %s %s %s %d', vol{1}, errors{1}, dist{1}, has_mean);
                % last: h_T of each draw, and e_T and u_T from y and the draw's mu, rho and psi
                pv = zeros(20, 7); pv(:, used) = r.params;
                assert(isequal(size(r.last), [20 3]) && isequal(r.last(4:4:20, 1), r.draws.h(:, end)), ...
                    'sv: last does not hold h_T, %s %s %s %d', vol{1}, errors{1}, dist{1}, has_mean);
                for i = 1:20
                    e = y - pv(i,1); u = filter([1 -pv(i,2)], [1 pv(i,3)], e);
                    assert(r.last(i,2) == e(T) && abs(r.last(i,3) - u(T)) < 1e-10*(1 + abs(u(T))), ...
                        'sv: last does not hold e_T and u_T, %s %s %s %d', vol{1}, errors{1}, dist{1}, has_mean);
                end
            end
        end
    end
end

% the default prior, given explicitly, changes nothing; a changed prior changes the draws
ra = ssm.models.sv(y, 'errors', 'ma', 'nsim', 20, 'burnin', 5, 'seed', 6);
rb = ssm.models.sv(y, 'errors', 'ma', 'nsim', 20, 'burnin', 5, 'seed', 6, 'prior', struct('Vpsi', 1));
rc = ssm.models.sv(y, 'errors', 'ma', 'nsim', 20, 'burnin', 5, 'seed', 6, 'prior', struct('Vpsi', 1 + 1e-9));
assert(isequal(ra.params, rb.params) && ~isequal(ra.params, rc.params), 'sv: the prior option does not act as documented');

% malformed input
bad = {{y, 'vol', 'ar2'}, {y, 'errors', 'ar'}, {y, 'dist', 'skewt'}, {y, 'nsim'}, {y, 'lags', 2}, ...
    {y, 'prior', struct('rho0', 0)}, {y, 'errors', 'ma', 'prior', struct('h0', 0)}, {[1; 2; NaN; 4; 5; 6]}, ...
    {y, 'nsim', 0}};
for k = 1:numel(bad)
    try
        ssm.models.sv(bad{k}{:});
        error('test:noError', 'case %d did not raise', k);
    catch err
        assert(startsWith(err.identifier, 'ssm:models:sv:'), 'sv: case %d raised %s', k, err.identifier);
    end
end
end
