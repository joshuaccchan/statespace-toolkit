function llike = intlike_sf(shortY,YmXb,A,invSig,invOmega)

% This function evaluates the log-integrated likelihood of the static
% factor model
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

[T n] = size(shortY);
q = size(A,2);
Tn = T*n;
AinvSig = A'*sparse(1:n,1:n,invSig);
invAApSig = sparse(1:n,1:n,invSig) ...
    - AinvSig'/(sparse(1:q,1:q,invOmega)+ AinvSig*A)*AinvSig;
CinvAApSig = sparse(chol(invAApSig,'lower'));
temp = sum(reshape(CinvAApSig'*reshape(YmXb,n,T),Tn,1).^2);
llike = -Tn/2*log(2*pi) + T*sum(log(diag(CinvAApSig))) -.5*temp;
end
