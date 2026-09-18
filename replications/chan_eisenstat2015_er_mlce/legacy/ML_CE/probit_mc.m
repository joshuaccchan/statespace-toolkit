% % =======================================================================
% % fit the mroz dataset with the probit model and compute the marginal
% % likelihood using the cross-entropy method
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
load mroz200w.txt
N = size( mroz200w,1);
Y = mroz200w(:,1);
X = [ones(N,1) mroz200w(:,2:end)];
X(:,5) = X(:,5)/100;
k = size(X,2);
nloop = 5500; 
burnin = 500;

% parameters for the prior
beta0 = zeros(k,1);
Vbeta = 10*eye(k)/(pi^2/3);
invVbeta = Vbeta\speye(k);

% set initial values
beta = zeros(k,1);
Yhat = Y/10;
id0 = find(Y==0); l0 = length(id0);
id1 = find(Y==1); l1 = length(id1);

store_beta = zeros(nloop-burnin,k);
invDb = invVbeta + X'*X;
start_time = clock;
rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

disp('Starting MCMC.... ');
disp(' ' );

% MCMC starts here
for loop = 1 : nloop
        % sample beta
    db = invVbeta * beta0 + X' * Yhat;
    beta = invDb\db + chol(invDb)\randn(k,1);
        % sample Yhat
    Ymu = X*beta;
    Yhat (id0) = tnormrnd( Ymu( id0, : ), ones(l0, 1 ), -inf, 0 );
    Yhat (id1) = tnormrnd( Ymu( id1, : ), ones(l1, 1 ), 0 , inf );   
    if loop > burnin
        store_beta( loop - burnin, : ) = beta';    
    end    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

%% compute ML using CE
ndraws = 5000;
nbigloop = 10;
probit_CE
ML = mean(sml)
MLstd = std(sml)/sqrt(nbigloop)



