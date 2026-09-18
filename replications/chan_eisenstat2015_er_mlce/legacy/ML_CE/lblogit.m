% % =======================================================================
% % support function: evaluate the log-likelihood of the logit model
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================
function f = lblogit(beta,Y,X)
f = sum((Y-1).*(X*beta)-log(1+exp(-X*beta)));
end
