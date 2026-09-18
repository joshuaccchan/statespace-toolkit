% % =======================================================================
% % Dynamic factor-VAR example -- compute the marginal likelihood using the 
% % cross-entropy method
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
load USdata.csv;
tempY = USdata;
Y0 = tempY(1,:)';
shortY = tempY(2:end,:);
[T n] = size(shortY);
q = 1;  % no. of factors
Tn = T*n;
nnp1 = n*(n+1);
r = n*(n+1);
nloop = 10500;
burnin = 500;
Y = reshape(shortY',Tn,1);

%% initialize
store_beta = zeros(nloop-burnin,n*(n+1));
store_A = zeros(nloop-burnin,n);
store_Sig1 = zeros(nloop-burnin,n);
store_Sig2 = zeros(nloop-burnin,1);
store_phi = zeros(nloop-burnin,1);
countphi=0;
countA = 0;

%% prior
a0 = sparse(n-1,1);
beta0 = sparse(r,1);
invVbeta = eye(n*(n+1))/10;
Vbeta0 = invVbeta\speye(r);
invVa = eye(n-1);
Va0 = invVa\speye(n-1);
nu01 = 6; S01 = ones(n,1);
nu02 = 6; S02 = 1;

%% construct a few thigns
X = [Y0; Y(1:end-n)];
bigG = zeros(T*n,n*(n+1));
for t=1:T
    bigG((t-1)*n+1:t*n,:) = kron(eye(n), [ 1 X((t-1)*n+1:t*n,:)']);
end

start_time = clock;
%% initialize the Markov chain
[beta f A Sig1 Sig2 phi] = initia_simple(Y,bigG,n);
invSig1 = diag(1./diag(Sig1));
invSig2 = 1/Sig2;
a = A(2:n);
invS = kron(speye(T,T), 1); invS(1,1) = 1-phi^2;
Ac = A;

newnu1 = nu01 + T;
newnu2 = nu02 + T;

rand('state', sum(100*clock) );
randn('state', sum(200*clock) );

disp('Starting MCMC.... ');
disp(' ' );
for loop = 1:nloop    
   
    %% sample beta marginal of f
    H = speye(T,T) - phi*[[sparse(q, (T-1)*q); kron(speye(T-1,T-1), speye(q))] sparse(T,q)];
    invS(1,1) = 1-phi^2;
    F0 = H'*invS*H;
    AinvSig1 = A'*invSig1;
    P = F0/Sig2 + speye(T)*(AinvSig1*A);
    Ytilde = (AinvSig1*reshape(Y,n,T))';
    bigGtilde = kron(speye(T), AinvSig1)*bigG;
    GinvSig1 = bigG'*kron(speye(T,T), invSig1);
    invDbeta = invVbeta + GinvSig1*bigG - bigGtilde'*(P\bigGtilde);
    dbeta = GinvSig1*Y - bigGtilde'*(P\Ytilde);
    beta = invDbeta\dbeta + chol(invDbeta)\randn(n*(n+1),1);
    bigGbeta = bigG*beta;
    
    %% sample A marginal of f
    bigA = kron(speye(T),A);
    AinvSig1 = bigA'*kron(speye(T,T), invSig1);
    AinvSig1A = AinvSig1*bigA;
    invP = F0/Sig2 + AinvSig1A;
    C = chol(invP);
    fhat = invP\(AinvSig1 * (Y-bigGbeta));
    fstar = fhat;
    [at ahat invDa] = maxA(fstar,Y,bigGbeta,invVa,invSig1,1,2:n); % obtain a proposal density
    ac = at + chol(invDa)\randn(n-1,1);
    Ac = [1; ac];    
    bigAc = kron(speye(T),Ac);
    AcinvSig1 = bigAc'*kron(speye(T,T), invSig1);
    AcinvSig1Ac = AcinvSig1*bigAc;
    invPc = F0/Sig2 + AcinvSig1Ac;
    Cc = chol(invPc);
    fhatc = invPc\(AcinvSig1 * (Y-bigGbeta));
    acerr = ac-ahat;    aerr = a-ahat;    
    lalp = -.5*acerr'*invDa*acerr+.5*aerr'*invDa*aerr +...
        +sum(log(diag(C))) - sum(log(diag(Cc))) + 1/2*(fstar-fhatc)'*invPc*(fstar-fhatc) +...
        -1/2*(a-at)'*invDa*(a-at) + 1/2*(ac-at)'*invDa*(ac-at); % MH acceptance prob   
    if exp(lalp)> rand
        a = ac;
        A = Ac;
        countA = countA+1;        
    end 
    
    %% sample f
    bigA = kron(speye(T),A);
    AinvSig1 = bigA'*kron(speye(T,T), invSig1);
    AinvSig1A = AinvSig1*bigA;
    invP = F0/Sig2 + AinvSig1A;
    C = chol(invP);
    fhat = invP\(AinvSig1 * (Y-bigGbeta));
    f = fhat + C\randn(T,1);   
    
    %% sample Sig1
    err1 = reshape(Y - bigG*beta-kron(f,A),n,T);
    newS1 = S01 +  sum(err1.^2,2);
    diaginvSig1 = gamrnd(newnu1/2, 2./newS1);
    invSig1 = diag(diaginvSig1);
    Sig1 = diag(1./diag(invSig1));
 
    %% sample Sig2
    err2 = reshape(H*f,q,T);
    newS2 = S02 + sum(err2(2:end).^2,2) + err2(1).^2*(1-phi^2);
    Sig2 = diag(1./gamrnd(newnu2/2, 2./newS2));
    
    %% sample phi
    tempsum = f(1:T-1)'*f(1:T-1);
    phihat = f(1:T-1)'*f(2:T)/tempsum;
    Vphi = Sig2/tempsum;
    phic = phihat + sqrt(Vphi)*randn;
    if abs(phic)<.999
        lalp = log(1-phic^2)/2 - (1-phic^2)/(2*Sig2)*f(1)^2 - log(1-phi^2)/2 + (1-phi^2)/(2*Sig2)*f(1)^2;
        if exp(lalp)> rand
            phi = phic;
            countphi = countphi+1;
        end
    end 

     if loop>burnin
        i = loop-burnin;
        store_beta(i,:) = beta;
        store_phi(i) = phi;
        store_A(i,:) = A;
        store_Sig1(i,:) = diag(Sig1)';
        store_Sig2(i,:) = diag(Sig2)';
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
DFVAR_CE
ML = mean(sml)
MLstd = std(sml)/sqrt(nbigloop)






