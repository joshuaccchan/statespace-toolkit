% ssm.models.sv_forecast - the predictive distribution of y_{T+1} at each posterior draw of
% ssm.models.sv. Written for this toolkit.
%
%   f = ssm.models.sv_forecast(res)
%
%   res : the output of ssm.models.sv, or any struct with its fields spec, params,
%         param_names and last
%   f.m      : nsim x 1 means of y_{T+1} given each draw, mu + rho*e_T + psi*u_T
%   f.h      : nsim x 1 draws of h_{T+1}, one per draw
%   f.nu     : nsim x 1 degrees of freedom, Inf under normal errors
%   f.v      : nsim x 1 variances of y_{T+1} given each draw, with h_{T+1} integrated out
%   f.logpdf : handle; f.logpdf(x) is the log predictive density at each element of x
%   f.cdf    : handle; f.cdf(x) is the predictive distribution function at each element of x
%
% Given a draw and h_{T+1}, y_{T+1} is normal with mean f.m and variance exp(h_{T+1}), or
% Student-t with f.nu degrees of freedom, location f.m and scale exp(h_{T+1}/2). The mean of
% h_{T+1} is a = mu_h + phi*(h_T - mu_h) under an AR(1) and a = h_T under a random walk, so
% f.v = exp(a + sig2_h/2), times nu/(nu - 2) under Student-t errors. The predictive density
% is the equally weighted mixture of the densities given f.h over the draws. Its mean is
% mean(f.m) and its variance mean(f.v) + var(f.m, 1). Random numbers: randn(nsim, 1).

function f = sv_forecast(res)
P = res.params; n = size(P, 1); m = res.spec;
col = @(name) P(:, strcmp(res.param_names, name));
hT = res.last(:,1); eT = res.last(:,2); uT = res.last(:,3);
if m.mean, mu = col('mu'); else, mu = zeros(n, 1); end
switch m.errors
    case 'iid',  f.m = mu;
    case 'ma',   f.m = mu + col('psi').*uT;
    case 'arma', f.m = mu + col('rho').*eT + col('psi').*uT;
end
s2 = col('sig2_h');
if strcmp(m.vol, 'ar1')
    muh = col('mu_h'); a = muh + col('phi').*(hT - muh);
else
    a = hT;
end
f.h = a + sqrt(s2).*randn(n, 1);
f.v = exp(a + s2/2);
is_t = strcmp(m.dist, 't');
if is_t
    f.nu = col('nu'); f.v = f.v.*f.nu./(f.nu - 2);
else
    f.nu = inf(n, 1);
end
mm = f.m; s = exp(f.h/2); nu = f.nu;
f.logpdf = @(x) logpdf(x, mm, s, nu, is_t);
f.cdf = @(x) cdf(x, mm, s, nu, is_t);
end

function l = logpdf(x, m, s, nu, is_t)
% the log of the mean over draws of the density at each element of x
x = x(:)'; z = (x - m)./s;
if is_t
    lw = gammaln((nu + 1)/2) - gammaln(nu/2) - .5*log(nu*pi) - log(s) ...
        - (nu + 1)/2.*log1p(z.^2./nu);
else
    lw = -.5*log(2*pi) - log(s) - .5*z.^2;
end
c = max(lw, [], 1);
l = (c + log(mean(exp(lw - c), 1)))';
end

function p = cdf(x, m, s, nu, is_t)
% the mean over draws of the distribution function at each element of x
x = x(:)'; z = (x - m)./s;
if is_t
    p = mean(tcdf(z, repmat(nu, 1, numel(x))), 1)';
else
    p = mean(normcdf(z), 1)';
end
end
