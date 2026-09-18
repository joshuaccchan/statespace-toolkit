% This is the main run file for estimating the VAR(1) model and computing
% the (observed-data) DIC
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J. C. C. and Grant, A. L. (2016). "Fast Computation of the Deviance
% Information Criterion for Latent Variable Models," Computational 
% Statistics and Data Analysis, 100, 847-859.
%
% This code comes without technical support of any kind.  It
% is expected to reproduce the results reported in the paper.
% Under no circumstances will the authors be held responsible for any use
% (or misuse) of this code in any way.

clear; clc;
R = 10;           % number of parallel chains
nloop = 11000;    % number of simulations for each chain
burnin = 1000;    % number of initial draws discarded

load USdata.csv;  % [GDP growth, TBill rate, unemployment rate, CPI rate]
Y0 = USdata(1:3,:);
shortY = USdata(4:end,:);
[T n] = size(shortY);
Y = reshape(shortY',T*n,1);
p = 1;            % number of AR lags
k = n+p*n^2;    
    %% prior 
invVbeta = sparse(1:k,1:k,ones(1,k))/5;
nu0 = n+3; S0 = eye(n);
prior = @(b,S) -.5*b'*invVbeta*b - (nu0+n+1)/2*log(det(S)) - .5*trace(S\S0);
    %% compute and define a few things
X = zeros(T,n*p);
for i=1:p
    X(:,(i-1)*n+1:i*n) = [Y0(3-i+1:end,:); shortY(1:T-i,:)];
end
bigX = SURform2([ones(T,1) X],n); 
newnu = T + nu0;
c = -T*n/2*log(2*pi);
    %% initialize for storage
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);

disp('Starting MCMC.... ');

start_time = clock;
for bigloop = 1:R

        %% initialize the chain
    beta = (bigX'*bigX)\(bigX'*Y) + .1*randn(k,1);
    err = reshape(Y - bigX*beta,n,T);
    Sig = err*err'/T;    
    invSig = Sig\speye(n);
    
        %% MCMC starts here

    for loop = 1:nloop  
        
            %% sample beta
        XinvSig = bigX'*kron(speye(T),invSig);
        XinvSigX = XinvSig*bigX;
        invDbeta = invVbeta + XinvSigX;
        betahat = invDbeta\(XinvSig*Y);    
        beta =  betahat + chol(invDbeta,'lower')'\randn(k,1);  

            %% sample Sig
        err = reshape(Y - bigX*beta,n,T);
        newS = S0 + err*err';
        Sig = iwishrnd(newS,newnu);
        invSig = Sig\speye(n);

        if loop>burnin
            i = loop-burnin;
            u = Y-bigX*beta;
            llike = c - T/2*log(det(Sig)) - .5*u'*kron(speye(T),invSig)*u;
            lpost = llike + prior(beta,Sig);
            store_llike(i) = llike;       
            store_lpost(i) = lpost;
        end    
        if ( mod( loop, 5000 ) ==0 )
            disp(  [ num2str( loop ) ' loops... ' ] )
        end 
    
    end
        %% compute DIC2
    [~, id] = max(store_lpost);
    DIC2 = -4*mean(store_llike) + 2*store_llike(id);
    store_DIC(bigloop) = DIC2;
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

DIC = mean(store_DIC);             % (observed-data) DIC
DICNSE = std(store_DIC)/sqrt(R);   % numerical standard error
