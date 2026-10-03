function r = fc_origin(repo, y, z, t, name, fc, sd)
% fc_origin - estimate one model on the data up to quarter t and score its forecasts
% of y(t+1) and of mean(y(t+1:t+4)): r = [point1 lpl1 point4 lpl4 lo4 hi4], the
% posterior predictive means, the logs of the predictive densities at the outcomes
% (NaN where a target lies beyond the data), and the 5th and 95th percentiles of the
% predictive distribution of the four-quarter average. z is the long-run expectation,
% aligned with y and NaN before its first quarter, which 'CCK' takes as its presample;
% the other models ignore it. fc.nsim draws are kept after fc.burnin.
rng(sd, 'twister');
switch name
    case 'ARbound'
        out = ckp_run(repo, y(1:t), fc.nsim + fc.burnin, fc.burnin, Inf, sd);
        f = ckp_forecast(out, y(t));
    case 'CCK'
        i0 = find(~isnan(z), 1);
        out = cck_sampler(y(i0+1:t), y(i0), z(i0+1:t), fc.nsim, fc.burnin);
        f = cck_forecast(out, y(t));
    case 'AR4'
        f = ar_model(y(1:t), 4, fc.nsim, fc.burnin);
    otherwise
        out = uc_sampler(y(1:t), uc_model(name), fc.nsim, fc.burnin, fc.nsim);
        f = uc_forecast(uc_model(name), out);
end
r = nan(1, 6);
r(1) = mean(f.m1); r(2) = score(y(t+1), f.m1, f.v1);
r(3) = mean(f.m4);
if t + 4 <= numel(y), r(4) = score(mean(y(t+1:t+4)), f.m4, f.v4); end
r(5) = mixquantile(.05, f.m4, f.v4); r(6) = mixquantile(.95, f.m4, f.v4);
end

function x = mixquantile(p, m, v)
% the p-quantile of the equally weighted mixture of N(m_i, v_i)
s = sqrt(v);
x = fzero(@(q) mean(normcdf((q - m)./s)) - p, [min(m - 8*s) max(m + 8*s)]);
end

function s = score(x, m, v)
% log of the mean over draws of the normal density at x
l = -.5*log(2*pi*v) - .5*(x - m).^2./v;
s = max(l) + log(mean(exp(l - max(l))));
end
