% % =======================================================================
% % support function: marginal likelihood estimation for the TVP-VAR 
% % using the cross-entropy method 
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

sml = zeros(nbigloop,1);
c = -Tn/2*log(2*pi);
for bigloop=1:nbigloop
    disp( ['The '  num2str(bigloop) 'th loop...' ] );
    slsum_CE = zeros(ndraws,1);
      % obtain parameters for the IS density
    Sig1hat = squeeze(mean(store_Sig1)); Sig2hat = mean(store_Sig2)'; 
    nu1hat = T;
    S1hat = nu1hat*(squeeze(mean(store_invSig1))\speye(n));
    nu2hat = zeros(q,1);    S2hat = zeros(q,1);
    for i=1:q
        temp = gamfit(1./store_Sig2(:,i));
        nu2hat(i) = temp(1); S2hat(i) = temp(2);    
    end
      % IS starts here
    for loop = 1:ndraws
        Sig1 = iwishrnd(S1hat, nu1hat); invSig1 = Sig1\speye(n);
        invSig2 = gamrnd(nu2hat,S2hat); Sig2 = 1./invSig2;
    
        invS = kron(speye(T), sparse(diag(invSig2))); invS(1:q,1:q) = invD;
        K = H'*invS*H;
        GinvSig1 = bigG'*kron(speye(T), invSig1);
        GinvSig1G = GinvSig1*bigG;
        invP = H'*invS*H + GinvSig1G;   C = chol(invP);
        betamu = invP\(GinvSig1 * Y);
        betahat = betamu;    
        err1 = Y-bigG*betahat;
        llike_CE = c - T/2*log(det(Sig1)) -.5*err1'*kron(speye(T),invSig1)*err1 + ...  % integrated likelihood
            + sum(log(diag(chol(K)))) - .5*betahat'*K*betahat - sum(log(diag(C)));   
        lpri_CE = linvwishpdf(Sig1,nu01,S01) + sum(linvgammpdf(Sig2,nu02/2,S02/2));    % prior density
        linst_CE = sum(linvgammpdf(Sig2,nu2hat,1./S2hat)) + ...
            + linvwishpdf(Sig1,nu1hat,S1hat);   % IS density
        slsum_CE(loop) =  llike_CE + lpri_CE - linst_CE;   
    end
    meanllike = mean(slsum_CE);
    MLIS = log(mean(exp(slsum_CE-meanllike))) + meanllike;
    sml(bigloop) = MLIS;
end
