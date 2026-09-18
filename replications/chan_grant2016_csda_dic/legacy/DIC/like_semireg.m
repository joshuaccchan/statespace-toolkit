function llike = like_semireg(ymXbeta,G,D,theta,sig2,tau,V1,V2)

% This function evaluates the log-complete-data likelihood of the
% semiparametric regression
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J. C. C. and Grant, A. L. (2016). "Fast Computation of the Deviance
% Information Criterion for Latent Variable Models," Computational 
% Statistics and Data Analysis, 100, 847-859.
%
% This code comes without technical support of any kind.  It
% is expected to reproduce the results reported in the paper.
% Under no circumstances will the authors be held responsible for any use
% (or misuse) of this code in any way.

[n m] = size(D);
invOmega = sparse(1:m,1:m,[1/V1 1/V2 1/tau*ones(1,m-2)]);
GinvOmegaG = G'*invOmega*G;
c = -(n+m)/2*log(2*pi) - .5*log(V1) - .5*log(V2);
u = ymXbeta-D*theta;
llike = c - n/2*log(sig2) + sum(log(diag(G))) - (m-2)/2*log(tau) ...
    - .5/sig2*(u'*u) -.5*theta'*GinvOmegaG*theta;
end

