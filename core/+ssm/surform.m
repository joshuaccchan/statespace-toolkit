% ssm.surform - T x Tk block-diagonal sparse SUR expansion, the design matrix of
% a regression whose coefficients are stacked over time.
%
%   Xout = ssm.surform(X)
%
%   X    : T x k matrix of regressors
%   Xout : T x T*k sparse matrix whose row t holds X(t,:) in columns
%          (t-1)*k + (1:k), so that Xout*[b_1; b_2; ...; b_T] returns the
%          fitted series of the TVP regression y_t = X(t,:)*b_t
%
% Xout is the matrix Z = diag(x_1',...,x_T') of the book's TVP regression. Code-
% identical to bvar.util.surform in bvar-toolkit, and the two must stay so
% (tests/unit/test_twins.m).
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.3.

function Xout = surform( X )
[r,c] = size( X );
idi = kron((1:r)',ones(c,1));
idj = (1:r*c)';
Xout = sparse(idi,idj,reshape(X',r*c,1));
end