% % =======================================================================
% % support function: evaluate the log-density of a multivariate t
% % distribution
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================
function den = lmtpdf(X,nu,mu,sigma)
d = length(mu);
K = gammaln((nu+d)/2)-gammaln(nu/2) - d/2*log(nu) - d/2*log(pi) - log(det(sigma))/2;
err = X-mu;
u = err'*(sigma\err);
kernel = -(nu+d)/2 * log(1+u/nu);
den = K + kernel;
end
