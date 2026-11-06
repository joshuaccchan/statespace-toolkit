% ssm.intlike_sv_t - integrated log likelihood of the stochastic volatility model with
% Student-t errors, the log-volatilities integrated out by importance sampling:
%   y_t - mu = exp(h_t/2)*eps_t,  eps_t ~ Student-t with nu degrees of freedom,
%   h_t = muh + phih*(h_{t-1} - muh) + v_t,  v_t ~ N(0,omegah2),
%   h_1 ~ N(muh, omegah2/(1-phih^2)),  with omegah2 a VARIANCE.
%
%   [llike, nse] = ssm.intlike_sv_t(s2, muh, phih, omegah2, nu, ht, M)
%   [llike, nse] = ssm.intlike_sv_t(..., 'Tol', 1e-4, 'MaxIter', 500)
%
%   s2        : T x 1 squared demeaned data, (y - mu).^2, finite and nonnegative
%   muh       : the mean of h, finite
%   phih      : the persistence of h, |phih| < 1
%   omegah2   : the innovation variance of h, positive and finite
%   nu        : degrees of freedom, positive and finite
%   ht        : T x 1 finite starting point of the search for the mode of p(h | y), for
%               example the posterior mean of h
%   M         : number of importance draws, a positive integer
%   'Tol'     : stop the EM algorithm, and each of its Newton-Raphson searches, once
%               max|step| <= Tol (default 1e-4)
%   'MaxIter' : raise if the EM algorithm has not stopped after this many iterations, or
%               one of its Newton-Raphson searches after this many steps (default 500)
%   llike     : log p(y | mu, muh, phih, omegah2, nu), the log of the mean of the M
%               importance weights
%   nse       : the numerical standard error of llike by the delta method, the standard
%               deviation of the weights over sqrt(M) times their mean
%
% The mode hhat of p(h | y) is found by EM, with the scales of the Student-t errors as
% the missing data; each M-step is a Newton-Raphson search. The importance density is
% N(hhat, Kh^{-1}), where Kh is the negative Hessian of log p(h | y) at hhat, and the
% weights use the exact Student-t likelihood. nse approximates the standard deviation of
% llike over repeated calls: at T = 137 and 1000, its median over 200 calls was 0.90 to
% 1.10 of that standard deviation.
% Invalid inputs raise ssm:intlike_sv_t:bad* errors. A Newton-Raphson step that returns
% NaN or Inf raises ssm:intlike_sv_t:nonFinite, and reaching MaxIter in either loop raises
% ssm:intlike_sv_t:notConverged with the loop and the last step size. Random numbers per
% call: randn(T,M), drawn at once, so memory grows with T*M.
%
% See:
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics: GARCH versus
% Stochastic Volatility, Energy Economics, 54: 182-189.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.

function [llike, nse] = intlike_sv_t(s2,muh,phih,omegah2,nu,ht,M,opts)
arguments
    s2
    muh
    phih
    omegah2
    nu
    ht
    M
    opts.Tol (1,1) double {mustBePositive} = 1e-4
    opts.MaxIter (1,1) double {mustBeInteger, mustBePositive} = 500
end
if ~(isnumeric(s2) && isreal(s2) && iscolumn(s2) && ~isempty(s2) && all(isfinite(s2)) && all(s2 >= 0))
    error('ssm:intlike_sv_t:badS2', 's2 must be a nonempty column of finite, nonnegative values');
end
if ~(isnumeric(muh) && isscalar(muh) && isreal(muh) && isfinite(muh))
    error('ssm:intlike_sv_t:badMuh', 'muh must be a finite real scalar');
end
if ~(isnumeric(phih) && isscalar(phih) && isreal(phih) && abs(phih) < 1)
    error('ssm:intlike_sv_t:badPhih', 'phih must be a real scalar with |phih| < 1');
end
if ~(isnumeric(omegah2) && isscalar(omegah2) && isreal(omegah2) && omegah2 > 0 && isfinite(omegah2))
    error('ssm:intlike_sv_t:badOmegah2', 'omegah2 must be a positive, finite scalar');
end
if ~(isnumeric(nu) && isscalar(nu) && isreal(nu) && nu > 0 && isfinite(nu))
    error('ssm:intlike_sv_t:badNu', 'nu must be a positive, finite scalar');
end
if ~(isnumeric(ht) && isreal(ht) && iscolumn(ht) && numel(ht) == numel(s2) && all(isfinite(ht)))
    error('ssm:intlike_sv_t:badHt', 'ht must be a finite column with numel(s2) = %d elements', numel(s2));
end
if ~(isnumeric(M) && isscalar(M) && isreal(M) && M >= 1 && M == fix(M) && isfinite(M))
    error('ssm:intlike_sv_t:badM', 'M must be a positive integer');
end

T = length(s2);
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
HiSHdeltah = HiSH*deltah;
errh_out = 1; iter_out = 0;
while errh_out> opts.Tol
    if iter_out >= opts.MaxIter
        error('ssm:intlike_sv_t:notConverged', ['the EM algorithm for the mode of h: ' ...
            'no convergence to Tol = %g in %d iterations (last max|step| = %g)'], opts.Tol, opts.MaxIter, errh_out);
    end
    iter_out = iter_out + 1;
        % E-step    
    Eilam = (nu+1)./(nu+s2./exp(ht));
    s2Eilam = s2.*Eilam;
        % M-step
    htt = ht;
    errh_in = 1; iter_in = 0;
    while errh_in> opts.Tol
        if iter_in >= opts.MaxIter
            error('ssm:intlike_sv_t:notConverged', ['the Newton-Raphson search of EM iteration %d: ' ...
                'no convergence to Tol = %g in %d steps (last max|step| = %g)'], iter_out, opts.Tol, opts.MaxIter, errh_in);
        end
        iter_in = iter_in + 1;
        eht = exp(htt);
        sieht = s2Eilam./eht; 
        fh = -.5 + .5*sieht;
        Gh = .5*sieht;
        Kh = HiSH + sparse(1:T,1:T,Gh);
        if ~all(isfinite(fh) & isfinite(Gh))
            error('ssm:intlike_sv_t:nonFinite', ['the Newton-Raphson search of EM iteration %d: ' ...
                'step %d produced NaN or Inf'], iter_out, iter_in);
        end
        newht = Kh\(fh+Gh.*htt+HiSHdeltah);
        if ~all(isfinite(newht))
            error('ssm:intlike_sv_t:nonFinite', ['the Newton-Raphson search of EM iteration %d: ' ...
                'step %d produced NaN or Inf'], iter_out, iter_in);
        end
        errh_in = max(abs(newht-htt));
        htt = newht;          
    end    
    errh_out = max(abs(ht-htt));
    ht = htt;
end
    % compute negative Hessian
Gh = (nu+1)/(2*nu)*(s2.*exp(ht))./((exp(ht)+s2/nu).^2);
Kh = HiSH + sparse(1:T,1:T,Gh);
CKh = chol(Kh,'lower');

c = -.5*(T*log(omegah2) - log(1-phih^2)) - sum(log(diag(CKh)));
store_llike = zeros(M,1);
bigh = repmat(ht,1,M) + CKh'\randn(T,M);
for i=1:M    
    h = bigh(:,i);    
    store_llike(i) = c -.5*(h-deltah)'*HiSH*(h-deltah) + deny_h(s2,h,nu) + ...
        + .5*(h-ht)'*Kh*(h-ht);
end
maxllike = max(store_llike);
llike = maxllike + log(mean(exp(store_llike-maxllike)));
w = exp(store_llike-maxllike);                  % the weights, scaled by their largest
nse = std(w)/(sqrt(M)*mean(w));                 % the delta method for the log of their mean
end

function llike = deny_h(s2,h,nu)
T = size(s2,1);
c = T*(gammaln((nu+1)/2) - gammaln(nu/2) - .5*log(nu*pi));
llike = c - .5*sum(h) -(nu+1)/2*sum(log(1+(s2./exp(h))/nu));
end
