% % =======================================================================
% % TVP-VAR example -- compute the marginal likelihood using the 
% % cross-entropy method
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

%% initialize for storage
store_Sig1 = zeros(nloop-burnin,n,n);
store_Sig2 = zeros(nloop-burnin,q);
store_beta = zeros(nloop-burnin,Tq);
store_invSig1 = zeros(nloop-burnin,n,n);

Sig1 = cov(USdata);
Sig2 = .01 * eye(q);

invSig1 = inv(Sig1);
invSig2 = inv(Sig2);
beta = zeros(Tq,1);

%% prior
nu01 = n+3; S01 = eye(n);
nu02 = 6; S02 = .01*ones(q,1);
D = 10*eye(q); invD = inv(D);

%% construct a few thigns
X = [Y0; Y(1:end-n)];
bigG = SURform([ones(n*T,1) kron(reshape(X ,n,T)',ones(n,1))]);
H = speye(Tq,Tq) - [ [ sparse(q, (T-1)*q); kron(speye(T-1,T-1), speye(q))] sparse(Tq, q)];
invS = kron(speye(T,T), invSig2); invS(1:q,1:q) = invD;
K = H'*invS*H;
newnu1 = nu01 + T;
newnu2 = nu02 + T - 1;

rand('state', sum(100*clock) );
randn('state', sum(200*clock) );

disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;

for loop = 1:nloop
    
    %% sample Sig1 (measurement equation error variance)
    err1 = reshape(Y - bigG*beta,n,T);
    newS1 = S01 + err1*err1';
    invSig1 = wishrnd(newS1\speye(n),newnu1);
    Sig1 = invSig1\speye(n);

    %% sample Sig2 (state equation error variance - diagonal)
    err2 = reshape(H*beta,q,T);
    newS2 = S02 + sum(err2(:,2:end).^2,2);
    diaginvSig2 = gamrnd(newnu2/2, 2./newS2);
    invSig2 = diag( diaginvSig2 );
    Sig2 = diag(1./diaginvSig2);

    %% sample beta
    invS = kron(speye(T), sparse(invSig2)); invS(1:q,1:q) = invD;
    GinvSig1 = bigG'*kron(speye(T,T), invSig1);
    GinvSig1G = GinvSig1*bigG;
    invP = H'*invS*H + GinvSig1G;
    C = chol(invP);
    betahat = invP\(GinvSig1 * Y);
    beta = betahat + C\randn(Tq,1);

     if loop>burnin
        i=loop-burnin;
        store_beta(i,:) = beta';
        store_Sig1(i,:,:) = Sig1;
        store_Sig2(i,:) = diag(Sig2)';     
        store_invSig1(i,:,:) = invSig1;
    end
    
     if ( mod( loop, 2000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end
    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );


%% compute ML using CE
ndraws = 10000;
nbigloop = 10;
TVPVAR_CE
ML = mean(sml)
MLstd = std(sml)/sqrt(nbigloop)


