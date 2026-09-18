% % =======================================================================
% % support function: marginal likelihood estimation for the DF-VAR 
% % using the cross-entropy method 
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

    % obtain parameters for the IS density
store_theta = [store_beta store_A(:,2:end)];
thetahat = mean(store_theta)';
Vtheta = cov(store_theta);
phihat = mean(store_phi)';
Vphi = var(store_phi)';
sqrtVphi = sqrt(Vphi);
c = -Tn/2*log(2*pi);

Sig1hat = zeros(n,2);
Sig2hat = zeros(q,2);
for i=1:n
    Sig1hat(i,:) = gamfit(1./store_Sig1(:,i));
end
for i=1:q
    Sig2hat(i,:) = gamfit(1./store_Sig2(:,i));
end
bigSig1 = zeros(ndraws,n); bigSig2 = zeros(ndraws,q);

    % IS starts here
sml = zeros(nbigloop,1);
for bigloop=1:nbigloop    
    bigtheta = mvnrnd(thetahat',Vtheta,ndraws);
    bigphi = tnormrnd(phihat,Vphi,-1,1,ndraws);
    for i=1:n
        bigSig1(:,i) = 1./gamrnd(Sig1hat(i,1),Sig1hat(i,2),ndraws,1);
    end
    for i=1:q
        bigSig2(:,i) = 1./gamrnd(Sig2hat(i,1),Sig2hat(i,2),ndraws,1);
    end
    A = ones(n,1); 
    shortY = reshape(Y,n,T)';
    slsum_CE = zeros(ndraws,1);
    for loop=1:ndraws
        theta = bigtheta(loop,:)';
        beta = theta(1:r);
        a = theta(r+1:end);
        phi = bigphi(loop,:)';
        dSig1 = bigSig1(loop,:)';
        dSig2 = bigSig2(loop,:)';
        dinvSig1 = 1./dSig1;
        dinvSig2 = 1./dSig2;
        invSig1 = diag(dinvSig1);
        invSig2 = diag(dinvSig2);
        spinvSig2 = sparse(invSig2);
        A(2:end) = a;        
        bigGbeta = bigG*beta;
        bigA = kron(speye(T),A);
    
        H = speye(T) - sparse(2:T,1:(T-1),phi*ones(1,T-1),T,T); 
        invS = kron(speye(T), invSig2); invS(1:q,1:q) = spdiags((1-phi.^2)./dSig2,0,q,q);
        K = H'*invS*H;
        AinvSig1 = bigA'*kron(speye(T), invSig1);
        AinvSig1A = AinvSig1*bigA;
        invP = K + AinvSig1A;
        C = chol(invP);
        fmu = invP\(AinvSig1*(Y-bigGbeta));
        fhat = fmu;    
        yerr = Y - bigGbeta - bigA*fhat;
        
        llike_CE = c - T/2*sum(log(dSig1)) -1/2*yerr'*kron(speye(T),invSig1)*yerr + ...
            + sum(log(diag(chol(K)))) - 1/2*fhat'*K*fhat - sum(log(diag(C)));  % integrated likelihood
        lpri_CE = lmvnpdf(beta,beta0,Vbeta0) + lmvnpdf(a,a0,Va0) - log(2) + ...
            + sum(linvgammpdf(dSig1,nu01/2,S01/2)) + sum(linvgammpdf(dSig2,nu02/2,S02/2)); % prior density
        linst_CE =lmvnpdf(theta,thetahat,Vtheta) + ...
            + log(normpdf(phi,phihat,sqrtVphi)/(normcdf((1-phihat)/sqrtVphi)-normcdf((-1-phihat)/sqrtVphi))) + ...
            + sum(linvgammpdf(dSig1,Sig1hat(:,1),1./Sig1hat(:,2)))+...
            + sum(linvgammpdf(dSig2,Sig2hat(:,1),1./Sig2hat(:,2)));  % IS density        
        slsum_CE(loop) = llike_CE + lpri_CE - linst_CE;
    end
    maxllike = max(slsum_CE);
    MLIS = log(mean(exp(slsum_CE-maxllike))) + maxllike;
    sml(bigloop) = MLIS;
end
