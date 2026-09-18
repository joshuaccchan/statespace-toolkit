% % =======================================================================
% % support function: initialize the Markov chain for a DF-VAR
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

function [beta f A Sig1 Sig2 phi] = initia_simple(Y,bigG,n)

T = length(Y)/n;

%% initialize
phi = .6*rand;
H = speye(T,T) - phi*[ [ sparse(1, T-1); speye(T-1,T-1)] sparse(T, 1)];
Sig1 = diag(var(reshape(Y,n,T)'));
invSig1 = diag(1./diag(Sig1));
invS = speye(T,T); invS(1,1) = 1-phi^2;
F0 = H'*invS*H;
Sig2 = mean(diag(Sig1));
A = [1; zeros(n-1,1)];

%% sample beta marginal of f
AinvSig1 = A'*invSig1;
P = F0/Sig2 + speye(T)*(AinvSig1*A);
Ytilde = (AinvSig1*reshape(Y,n,T))';
bigGtilde = kron(speye(T), AinvSig1)*bigG;
GinvSig1 = bigG'*kron(speye(T,T), invSig1);
invDbeta = GinvSig1*bigG - bigGtilde'*(P\bigGtilde);
dbeta = GinvSig1*Y - bigGtilde'*(P\Ytilde);
beta = invDbeta\dbeta;
bigGbeta = bigG*beta;

bigA = kron(speye(T),A);
AinvSig1 = bigA'*kron(speye(T,T), invSig1);
AinvSig1A = AinvSig1*bigA;
invP = F0/Sig2 + AinvSig1A;
f = invP\(AinvSig1 * (Y-bigGbeta));
for i=1:100
    %% construct K
    H = speye(T,T) - phi*[ [ sparse(1, (T-1)*1); speye(T-1,T-1)] sparse(T, 1)];
    invS(1,1) = 1-phi^2;
    F0 = H'*invS*H;
    %% sample beta marginal of f
    AinvSig1 = A'*invSig1;
    P = F0/Sig2 + speye(T)*(AinvSig1*A);
    Ytilde = (AinvSig1*reshape(Y,n,T))';
    bigGtilde = kron(speye(T), AinvSig1)*bigG;
    GinvSig1 = bigG'*kron(speye(T,T), invSig1);
    invDbeta =  GinvSig1*bigG - bigGtilde'*(P\bigGtilde);
    dbeta = GinvSig1*Y - bigGtilde'*(P\Ytilde);
    beta = invDbeta\dbeta + chol(invDbeta)\randn(n*(n+1),1);
    bigGbeta = bigG*beta;
      
     %% sample f
    bigA = kron(speye(T),A);
    AinvSig1 = bigA'*kron(speye(T,T), invSig1);
    AinvSig1A = AinvSig1*bigA;
    invP = F0/Sig2 + AinvSig1A;
    C = chol(invP);
    fhat = invP\(AinvSig1 * (Y-bigGbeta));
    f = fhat + C\randn(T,1);   
    
    %% sample A
    bigf = kron(f,[sparse(1,n-1); speye(n-1)]);
    finvSig1 = bigf'*kron(speye(T,T), invSig1);
    invDa = finvSig1*bigf;
    da = finvSig1*(Y-bigGbeta - reshape([f zeros(T,n-1)]',T*n,1) );
    A(2:end) = invDa\da + chol(invDa)\randn(n-1,1);
    
    %% sample Sig1
    err1 = reshape(Y - bigGbeta-kron(f,A),n,T);
    newS1 =  sum(err1.^2,2);
    diaginvSig1 = gamrnd(T/2, 2./newS1);
    invSig1 = diag(diaginvSig1);
    Sig1 = diag(1./diag(invSig1));
 
    %% sample Sig2
    err2 = reshape(H*f,1,T);
    newS2 = sum(err2(2:end).^2,2) + err2(1).^2*(1-phi^2);
    Sig2 = diag(1./gamrnd(T/2, 2./newS2));
    
    %% sample phi
    tempsum = f(1:T-1)'*f(1:T-1);
    phihat = f(1:T-1)'*f(2:T)/tempsum;
    Vphi = Sig2 / tempsum;
    phic = phihat + sqrt(Vphi)*randn;
    if abs(phic)<.999
        lalp = log(1-phic^2)/2 - (1-phic^2)/(2*Sig2)*f(1)^2 - log(1-phi^2)/2 + (1-phi^2)/(2*Sig2)*f(1)^2;
        if exp(lalp)> rand
            phi = phic;         
        end
    end     
end
