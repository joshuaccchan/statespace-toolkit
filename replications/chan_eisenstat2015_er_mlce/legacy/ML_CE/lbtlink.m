% % =======================================================================
% % support function: evaluate the log-likelihood of the t-link model
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================
function f = lbtlink(beta,Y,X,nu)
muy = tcdf(X*beta,nu);
f = Y'*log(muy) + (1-Y)'*(log(1-muy));
end
