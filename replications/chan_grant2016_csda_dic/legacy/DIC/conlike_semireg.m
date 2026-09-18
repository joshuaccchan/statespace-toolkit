function llike = conlike_semireg(ymXbeta,D,theta,sig2)

% This function evaluates the log-conditional likelihood of the
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

n = size(D,1);
u = ymXbeta-D*theta;
llike = -n/2*log(2*pi*sig2) - .5/sig2*(u'*u);
end
