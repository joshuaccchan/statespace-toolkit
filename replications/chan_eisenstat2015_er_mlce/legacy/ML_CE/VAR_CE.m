% % =======================================================================
% % support function: marginal likelihood estimation for the VAR model
% % using the cross-entropy method 
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

function mlIS = VAR_CE(beta,invSig1,invVbeta,Y,bigX,nu01,S01,M)
n = size(invSig1,2);
Tn = length(Y);
T = Tn/n;
q = n*(n+1); 
betahat = mean(beta)'; betavar = cov(beta);
sllike = zeros(M,1);
sbetac = repmat(betahat',M,1) + (chol(betavar)' * randn(q,M))';
newnu = T;
newS1 = newnu*(squeeze(mean(invSig1))\speye(n));
invnewS1 = newS1\speye(n);
beta0 = zeros(q,1); Vbeta0 = invVbeta\speye(q);
for loop = 1:M
    beta = sbetac(loop,:)';
    invSig1 = wishrnd(invnewS1,T);
    Sig1 = invSig1\speye(n);
    err1 = Y-bigX*beta;
    sllike(loop) = -Tn/2*log(2*pi) - T/2*log(det(Sig1)) - 1/2*err1'*kron(speye(T),invSig1)*err1 + ...
        + linvwishpdf(Sig1,nu01,S01) + lmvnpdf(beta,beta0,Vbeta0) + ...
        - linvwishpdf(Sig1,T,newS1) - lmvnpdf(beta,betahat,betavar);
end
meanllike = max(sllike);
mlIS = log(mean(exp(sllike-meanllike))) + meanllike;
end
