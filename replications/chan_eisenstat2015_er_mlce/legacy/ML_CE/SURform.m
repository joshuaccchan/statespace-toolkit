% % =======================================================================
% % support function: transform the multiple regression into SUR form
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

function Xout = SURform( X )
[r c] = size( X );
idi = kron((1:r)',ones(c,1));
idj = (1:r*c)';
Xout = sparse(idi,idj,reshape(X',r*c,1));
