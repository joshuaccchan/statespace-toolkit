function llike = intlike_tvpvar(Y,bigX,Sig,Omega,Q0,b0)

% This function evaluates the log-integrated likelihood of the TVP-VAR
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
T = length(Y)/n;
Tq = size(bigX,2);
q = Tq/T;
H = speye(Tq,Tq) - sparse(q+1:Tq,1:(T-1)*q,ones(1,(T-1)*q),Tq,Tq);
c = -T*n/2*log(2*pi)-.5*sum(log(Q0));
alp = H\[b0;sparse((T-1)*q,1)];
invOmega = 1./Omega;
invSig = Sig\speye(n);
XinvSig = bigX'*kron(speye(T),invSig);
XinvSigX = XinvSig*bigX;
HinvSH = H'*sparse(1:Tq,1:Tq,[1./Q0; repmat(invOmega,T-1,1)]')*H;
Kbeta = HinvSH + XinvSigX;
C = chol(Kbeta,'lower');
dbeta = HinvSH*alp + XinvSig*Y;
llike = c-(T-1)/2*sum(log(Omega))-T/2*log(det(Sig))-sum(log(diag(C))) + ...
    -.5*(Y'*kron(speye(T),invSig)*Y + alp'*HinvSH*alp - dbeta'*(Kbeta\dbeta));
end
