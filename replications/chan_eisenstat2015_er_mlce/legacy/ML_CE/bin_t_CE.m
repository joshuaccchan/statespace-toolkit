% % =======================================================================
% % support function: marginal likelihood estimation for the t-link model
% % using the cross-entropy method 
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================
	

sml = zeros(nbigloop,1);
slsum_CE = zeros(ndraws,1);
% optimize betat and Vbetat
betat = mean(store_beta)';
Vbetat = cov(store_beta);
for bigloop = 1:nbigloop
    sbetacan = mvnrnd(betat,Vbetat, ndraws )';
    for i = 1 : ndraws
        betacan = sbetacan( :, i );    
        llike_CE = lbtlink(betacan, Y, X, lamnu);   % t-link likelihood
        lpri_CE = lmvnpdf( betacan, beta0, Vbeta ); % prior density
        linst_CE = lmvnpdf( betacan, betat, Vbetat ); % IS density
        slsum_CE( i, 1 ) =  llike_CE + lpri_CE - linst_CE;    
    end
    maxllike = max(slsum_CE);
    MLIS = log(mean(exp(slsum_CE-maxllike))) + maxllike;
    sml(bigloop) = MLIS;
end
