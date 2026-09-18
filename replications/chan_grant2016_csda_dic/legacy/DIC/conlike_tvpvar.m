function llike = conlike_tvpvar(Y,bigX,beta,Sig)

% This function evaluates the log-conditional likelihood of the TVP-VAR
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

n = size(Sig,2);
Tn = length(Y);
T = Tn/n;
invSig = Sig\speye(n);
u = Y-bigX*beta;
llike = -Tn/2*log(2*pi) - T/2*log(det(Sig)) + ...
    -.5*u'*kron(speye(T),invSig)*u;    
end
