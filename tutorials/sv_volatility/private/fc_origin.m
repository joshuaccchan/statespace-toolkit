function r = fc_origin(y, t, W, name, fc, sd)
% fc_origin - estimate one model on the W returns y(t-W+1:t), or on all the returns y(1:t)
% when W = Inf, and forecast y(t+1): r = [m v lpl q01 q05 q95 q99], the mean and variance
% of the predictive distribution, the log of the predictive density at y(t+1) (NaN when
% t = numel(y)), and the 1st, 5th, 95th and 99th percentiles of the predictive
% distribution. The stochastic volatility models keep fc.nsim draws after fc.burnin, from
% a chain seeded with sd; 'EWMA' needs no draws.
yw = y(max(1, t-W+1):t);
if strcmp(name, 'EWMA')
    m = 0; v = ewma(yw, .94); s = sqrt(v);
    lp = @(x) -.5*log(2*pi*v) - .5*(x/s).^2;
    F = @(x) normcdf(x/s);
else
    o = sv_options(name);
    res = ssm.models.sv(yw, o{:}, 'nsim', fc.nsim, 'burnin', fc.burnin, 'seed', sd);
    f = ssm.models.sv_forecast(res);
    m = mean(f.m); v = mean(f.v) + var(f.m, 1);
    lp = f.logpdf; F = f.cdf;
end
r = nan(1, 7);
r(1:2) = [m v];
if t < numel(y), r(3) = lp(y(t+1)); end
p = [.01 .05 .95 .99];
for j = 1:4, r(3+j) = fzero(@(q) F(q) - p(j), m + 50*sqrt(v)*[-1 1]); end
end
