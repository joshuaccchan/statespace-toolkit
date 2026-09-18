% % =======================================================================
% % support function: marginal likelihood estimation for the probit model
% % using the cross-entropy method 
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

% optimize betat and Vbetat
betat = mean(store_beta)';
Vbetat = cov(store_beta);
sml = zeros(nbigloop,1);
slsum_CE = zeros(ndraws,1);
for bigloop = 1:nbigloop
    sbetacan = mvnrnd(betat,Vbetat,ndraws)';    
    for i = 1 : ndraws
        betacan = sbetacan(:,i);    
        llike_CE = lbprobit( betacan, Y, X );        % probit likelihood
        lpri_CE = lmvnpdf( betacan, beta0, Vbeta );  % prior density
        linst = lmvnpdf( betacan, betat, Vbetat );   % IS density
        slsum_CE(i) = llike_CE + lpri_CE - linst;
    end
    maxllike = max(slsum_CE);
    MLIS = log(mean(exp(slsum_CE-maxllike))) + maxllike;
    sml(bigloop) = MLIS;
end
