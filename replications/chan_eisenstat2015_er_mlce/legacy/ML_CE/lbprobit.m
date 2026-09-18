% % =======================================================================
% % support function: evaluate the log-likelihood of the probit model
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================
function f = lbprobit(beta,Y,X)
phi_muy = normcdf(X*beta);
f = Y'*log(phi_muy) + (1-Y)'*(log(1-phi_muy));
end
