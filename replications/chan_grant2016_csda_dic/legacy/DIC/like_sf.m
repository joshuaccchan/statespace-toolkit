function llike = like_sf(shortY,YmXb,A,shortf,invSig,invOmega) 

% This function evaluates the log-complete-data likelihood of the static
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
q = size(shortf,2);
Tn = T*n;
Tq = T*q;
f = reshape(shortf',Tq,1); 
c = -(Tn+Tq)/2*log(2*pi);
err = reshape(reshape(YmXb,n,T)-A*shortf',Tn,1);
llike = c + T/2*sum(log(invOmega)) -.5*f'*sparse(1:Tq,1:Tq,repmat(invOmega,T,1))*f ...
    + T/2*sum(log(invSig)) -.5*err'*sparse(1:Tn,1:Tn,repmat(invSig,T,1))*err; 
end
