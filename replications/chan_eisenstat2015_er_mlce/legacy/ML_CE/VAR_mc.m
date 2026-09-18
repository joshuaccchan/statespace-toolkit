% % =======================================================================
% % VAR example -- compute the marginal likelihood using the cross-entropy
% % method
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
load USdata.csv;
[T n] = size(USdata);
Y0 = USdata(1,:)';
Y = reshape(USdata(2:end,:)',(T-1)*n,1);
T = T-1;
q = n*(n+1);  % dim of states
Tq = T*q; Tn = T*n; q2 = q^2; nnp1 = n*(n+1);
nloop = 5500; burnin = 500;

%% initialize
store_beta = zeros(nloop-burnin,q);
store_Sig1 = zeros(nloop-burnin,n,n);
store_invSi1 = zeros(nloop-burnin,n,n);
Sig1 = cov(USdata);
invSig1 = Sig1\speye(n);
beta = zeros(q,1);

%% prior
invVbeta = speye(q)/10;
nu01 = n+3; S01 = eye(n);

%% construct a few thigns
X = reshape([Y0; Y(1:end-n)], n,T)';
bigX = zeros(n*T,q);
for t=1:T
    bigX((t-1)*n+1:t*n,:) = SURform([ones(n,1) kron(X(t,:),ones(n,1))]);
end
bigX = sparse(bigX);
newnu1 = T + nu01;

start_time = clock;
rand('state', sum(100*clock) );
randn('state', sum(200*clock) );

disp('Starting MCMC.... ');
disp(' ' );

for loop = 1:nloop
    
    %% sample beta|y,Sig1, Sig2
    XinvSig1 = bigX'*kron(speye(T), invSig1);
    XinvSig1X = XinvSig1*bigX;
    invP = invVbeta + XinvSig1X;
    C = chol(invP);
    betahat = invP\(XinvSig1 * Y);
    beta = betahat + C\randn(q,1);
    
    %% sample Sig1
    err1 = reshape(Y - bigX*beta,n,T);
    newS1 = S01 + err1*err1';
    invSig1 = wishrnd(inv(newS1),newnu1);
    Sig1 = inv(invSig1);
    
    if loop>burnin
        i=loop-burnin;
        store_beta(i,:) = beta';
        store_Sig1(i,:,:) = Sig1;
        store_invSi1(i,:,:) = invSig1;
    end
    
     if ( mod( loop, 2000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end
    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );


%% compute ML using CE
ndraws = 5000;
nbigloop = 10;
sml = zeros(nbigloop,1);
for bigloop = 1:nbigloop    
    sml(bigloop) = VAR_CE(store_beta,store_invSi1,invVbeta,Y,bigX,nu01,S01,5000);
end
ML = mean(sml)
MLstd = std(sml)/sqrt(nbigloop)
