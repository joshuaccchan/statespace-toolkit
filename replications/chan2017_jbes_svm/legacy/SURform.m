% This function constructs the regression matrix X so that the regression
% is in the SUR form
% 
% see: 
% Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with
% Time-Varying Parameters: An Application to Inflation Modeling, 
% Journal of Business and Economic Statistics, 35(1), 17-28.

function Xout = SURform(X)
[r c] = size( X );
idi = kron((1:r)',ones(c,1));
idj = (1:r*c)';
Xout = sparse(idi,idj,reshape(X',r*c,1));
end