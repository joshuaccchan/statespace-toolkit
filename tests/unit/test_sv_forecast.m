function test_sv_forecast
% ssm.models.sv_forecast must give, under each kind of log-volatility, errors and
% distribution, the means, variances and log-volatilities of its header; variances that
% match y_{T+1} simulated forward from the same draws; the log density of the mixture over
% the draws; and a distribution function that matches the simulated values. It must use one
% randn(nsim, 1) and run on the output of ssm.models.sv.

rng(11, 'twister'); n = 400;
P = [.1 + .05*randn(n,1), .5 + .1*rand(n,1), -.3 + .1*rand(n,1), -1 + .1*randn(n,1), ...
    .9 + .05*rand(n,1), .05 + .02*rand(n,1), 6 + 6*rand(n,1)];     % mu rho psi mu_h phi sig2_h nu
last = [-1 + .3*randn(n,1), .5*randn(n,1), .5*randn(n,1)];         % h_T e_T u_T
cases = {'ar1', 'iid', 'normal', true; 'ar1', 'ma', 't', true; 'rw', 'arma', 'normal', true; ...
         'rw', 'iid', 't', true; 'ar1', 'arma', 't', false};
for c = 1:size(cases, 1)
    [vol, errors, dist, has_mean] = cases{c,:};
    res = make(vol, errors, dist, has_mean, P, last);
    lab = sprintf('%s %s %s %d', vol, errors, dist, has_mean);
    mu = P(:,1)*has_mean; rho = P(:,2)*strcmp(errors, 'arma'); psi = P(:,3)*~strcmp(errors, 'iid');
    if strcmp(vol, 'ar1'), a = P(:,4) + P(:,5).*(last(:,1) - P(:,4)); else, a = last(:,1); end
    is_t = strcmp(dist, 't');

    % the means, the log-volatilities from one randn(n, 1), and the variances of the header
    rng(5, 'twister'); f = ssm.models.sv_forecast(res); nxt = rand;
    rng(5, 'twister'); z = randn(n,1);
    assert(rand == nxt, 'sv_forecast: it does not use exactly randn(nsim, 1), %s', lab);
    assert(max(abs(f.m - (mu + rho.*last(:,2) + psi.*last(:,3)))) < 1e-12, 'sv_forecast: the means differ, %s', lab);
    assert(max(abs(f.h - (a + sqrt(P(:,6)).*z))) < 1e-12, 'sv_forecast: h_{T+1} differs, %s', lab);
    v = exp(a + P(:,6)/2);
    if is_t, v = v.*P(:,7)./(P(:,7) - 2); assert(isequal(f.nu, P(:,7)), 'sv_forecast: nu differs, %s', lab);
    else, assert(all(isinf(f.nu)), 'sv_forecast: nu is not Inf, %s', lab); end
    assert(max(abs(f.v./v - 1)) < 1e-12, 'sv_forecast: the variances differ from the header, %s', lab);

    % the variances against y_{T+1} - f.m simulated from the same draws: the mean ratio of
    % the simulated to the stated variance is within four standard errors of 1
    R = 4000; rng(20 + c, 'twister');
    hs = a + sqrt(P(:,6)).*randn(n, R);
    if is_t, lam = 1./gamrnd(repmat(P(:,7)/2, 1, R), repmat(2./P(:,7), 1, R)); else, lam = 1; end
    e2 = lam.*exp(hs).*randn(n, R).^2;
    ratio = mean(e2, 2)./f.v; se = std(e2, 0, 2)./sqrt(R)./f.v;
    assert(abs(mean(ratio) - 1) < 4*sqrt(sum(se.^2))/n, ...
        'sv_forecast: the variances differ from the simulated ones, %s (%.4f)', lab, mean(ratio));

    % the log density of the mixture over the draws
    x = [-3; -.4; 0; .7; 5];
    s = exp(f.h/2);
    if is_t, d = tpdf((x' - f.m)./s, repmat(f.nu, 1, 5))./s;
    else, d = exp(-.5*((x' - f.m)./s).^2)./(sqrt(2*pi)*s); end
    assert(max(abs(f.logpdf(x) - log(mean(d, 1))')) < 1e-10, 'sv_forecast: the log density differs, %s', lab);

    % the distribution function against y_{T+1} simulated forward from the same draws, at
    % the 5th, ..., 95th percentiles of the simulated values
    rng(40 + c, 'twister'); R = 50; p = (.05:.05:.95)';
    Fq = zeros(numel(p), R); ys = zeros(n, R);
    for r = 1:R
        fr = ssm.models.sv_forecast(res);
        hr = a + sqrt(P(:,6)).*randn(n,1);
        if is_t, lr = 1./gamrnd(P(:,7)/2, 2./P(:,7)); else, lr = 1; end
        ys(:,r) = mu + rho.*last(:,2) + psi.*last(:,3) + sqrt(lr).*exp(hr/2).*randn(n,1);
        Fr{r} = fr; %#ok<AGROW>
    end
    q = quantile(ys(:), p);
    for r = 1:R, Fq(:,r) = Fr{r}.cdf(q); end
    dmax = max(abs(mean(Fq, 2) - p));
    assert(dmax < .01, 'sv_forecast: the distribution function differs from simulated values, %s (%.4f)', lab, dmax);
end

% the output of ssm.models.sv
rng(3, 'twister'); y = .2 + exp(.5*randn(200,1)/2).*randn(200,1);
for k = {{'vol', 'rw', 'errors', 'ma'}, {'errors', 'arma', 'dist', 't', 'mean', false}}
    r = ssm.models.sv(y, k{1}{:}, 'nsim', 50, 'burnin', 10, 'seed', 4);
    f = ssm.models.sv_forecast(r);
    assert(isequal(size(f.m), [50 1]) && all(isfinite([f.m; f.h; f.v; f.logpdf([0; 1]); f.cdf([0; 1])])), ...
        'sv_forecast: it does not run on the output of ssm.models.sv');
end
end

function res = make(vol, errors, dist, has_mean, P, last)
% the fields of the output of ssm.models.sv that ssm.models.sv_forecast reads
names = {'mu', 'rho', 'psi', 'mu_h', 'phi', 'sig2_h', 'nu'};
used = [has_mean, strcmp(errors, 'arma'), ~strcmp(errors, 'iid'), strcmp(vol, 'ar1'), ...
    strcmp(vol, 'ar1'), true, strcmp(dist, 't')];
res.params = P(:, used); res.param_names = names(used); res.last = last;
res.spec = struct('vol', vol, 'errors', errors, 'dist', dist, 'mean', has_mean);
end
