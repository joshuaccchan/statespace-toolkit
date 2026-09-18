function Xout = SURform2( X, n )

% This function constructs a sparse matrix Xout such that 
% for i=1:r
%    bigX((i-1)*n+1:i*n,:) = kron(speye(n),X(i,:));
% end
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

repX = kron(X,ones(n,1));
[r c] = size( X );
idi = kron((1:r*n)',ones(c,1));
idj = repmat((1:n*c)',r,1);
Xout = sparse(idi,idj,reshape(repX',n*r*c,1));
end
