% ssm.intlike_sv - integrated log likelihood of the stochastic volatility model, the
% log-volatilities integrated out by importance sampling:
%   y_t - mu = exp(h_t/2)*eps_t,  eps_t ~ N(0,1),
%   h_t = muh + phih*(h_{t-1} - muh) + v_t,  v_t ~ N(0,omegah2),
%   h_1 ~ N(muh, omegah2/(1-phih^2)),  with omegah2 a VARIANCE.
%
%   [llike, nse] = ssm.intlike_sv(s2, muh, phih, omegah2, ht, M)
%   [llike, nse] = ssm.intlike_sv(..., 'Tol', 1e-4, 'MaxIter', 500)
%
%   s2        : T x 1 squared demeaned data, (y - mu).^2, finite and nonnegative
%   muh       : the mean of h, finite
%   phih      : the persistence of h, |phih| < 1
%   omegah2   : the innovation variance of h, positive and finite
%   ht        : T x 1 finite starting point of the Newton-Raphson search for the mode of
%               p(h | y), for example the posterior mean of h
%   M         : number of importance draws, a positive integer
%   'Tol'     : stop the search once max|step| <= Tol (default 1e-4)
%   'MaxIter' : raise if that has not happened after this many steps (default 500)
%   llike     : log p(y | mu, muh, phih, omegah2), the log of the mean of the M
%               importance weights
%   nse       : the numerical standard error of llike by the delta method, the standard
%               deviation of the weights over sqrt(M) times their mean
%
% The importance density is N(hhat, Kh^{-1}), where hhat is the mode of p(h | y) and Kh
% the negative Hessian of log p(h | y) at the iterate before hhat. The weights use the
% exact Gaussian likelihood, so no mixture approximation enters. nse approximates the
% standard deviation of llike over repeated calls. With heavy-tailed weights it understates
% it: at T = 1000, its median over 200 calls was 0.58 to 0.83 of that standard deviation.
% Invalid inputs raise ssm:intlike_sv:bad* errors. A step that returns NaN or Inf raises
% ssm:intlike_sv:nonFinite, and reaching MaxIter raises ssm:intlike_sv:notConverged with
% the last step size. Random numbers per call: randn(T,M), drawn at once, so memory grows
% with T*M.
%
% See:
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics: GARCH versus
% Stochastic Volatility, Energy Economics, 54: 182-189.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.

function [llike, nse] = intlike_sv(s2,muh,phih,omegah2,ht,M,opts)
arguments
    s2
    muh
    phih
    omegah2
    ht
    M
    opts.Tol (1,1) double {mustBePositive} = 1e-4
    opts.MaxIter (1,1) double {mustBeInteger, mustBePositive} = 500
end
if ~(isnumeric(s2) && isreal(s2) && iscolumn(s2) && ~isempty(s2) && all(isfinite(s2)) && all(s2 >= 0))
    error('ssm:intlike_sv:badS2', 's2 must be a nonempty column of finite, nonnegative values');
end
if ~(isnumeric(muh) && isscalar(muh) && isreal(muh) && isfinite(muh))
    error('ssm:intlike_sv:badMuh', 'muh must be a finite real scalar');
end
if ~(isnumeric(phih) && isscalar(phih) && isreal(phih) && abs(phih) < 1)
    error('ssm:intlike_sv:badPhih', 'phih must be a real scalar with |phih| < 1');
end
if ~(isnumeric(omegah2) && isscalar(omegah2) && isreal(omegah2) && omegah2 > 0 && isfinite(omegah2))
    error('ssm:intlike_sv:badOmegah2', 'omegah2 must be a positive, finite scalar');
end
if ~(isnumeric(ht) && isreal(ht) && iscolumn(ht) && numel(ht) == numel(s2) && all(isfinite(ht)))
    error('ssm:intlike_sv:badHt', 'ht must be a finite column with numel(s2) = %d elements', numel(s2));
end
if ~(isnumeric(M) && isscalar(M) && isreal(M) && M >= 1 && M == fix(M) && isfinite(M))
    error('ssm:intlike_sv:badM', 'M must be a positive integer');
end

T = length(s2);
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
HiSHdeltah = HiSH*deltah;
errh = 1; iter = 0;
while errh> opts.Tol
    if iter >= opts.MaxIter
        error('ssm:intlike_sv:notConverged', ['the Newton-Raphson search for the mode of h: ' ...
            'no convergence to Tol = %g in %d steps (last max|step| = %g)'], opts.Tol, opts.MaxIter, errh);
    end
    iter = iter + 1;
    expht = exp(ht);
    sinvexpht = s2./expht;        
    fh = -.5 + .5*sinvexpht;
    Gh = .5*sinvexpht;
    Kh = HiSH + spdiags(Gh,0,T,T);    
    if ~all(isfinite(fh) & isfinite(Gh))
        error('ssm:intlike_sv:nonFinite', ['the Newton-Raphson search for the mode of h: ' ...
            'step %d produced NaN or Inf'], iter);
    end
    newht = Kh\(fh+Gh.*ht+HiSHdeltah);
    if ~all(isfinite(newht))
        error('ssm:intlike_sv:nonFinite', ['the Newton-Raphson search for the mode of h: ' ...
            'step %d produced NaN or Inf'], iter);
    end
    errh = max(abs(newht-ht));
    ht = newht;
end 
cholKh = chol(Kh,'lower');
c = -T/2*log(2*pi) -.5*(T*log(omegah2) - log(1-phih^2)) - sum(log(diag(cholKh)));
store_llike = zeros(M,1);
bigh = repmat(ht,1,M) + cholKh'\randn(T,M);
for i=1:M    
    h = bigh(:,i);    
    store_llike(i) = c - .5*(h-deltah)'*HiSH*(h-deltah) ...
        - .5*sum(h) - .5*exp(-h)'*s2 + ...
        + .5*(h-ht)'*Kh*(h-ht);
end
maxllike = max(store_llike);
llike = maxllike + log(mean(exp(store_llike-maxllike)));
w = exp(store_llike-maxllike);                  % the weights, scaled by their largest
nse = std(w)/(sqrt(M)*mean(w));                 % the delta method for the log of their mean
end
