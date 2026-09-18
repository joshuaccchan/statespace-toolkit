% % =======================================================================
% % fit the mroz dataset with the logit model and compute the marginal
% % likelihood using the cross-entropy method
% %
% % See Chan, J.C.C. and Eisenstat, E. (2013). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, forthcoming.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
load mroz200w.txt
N = size(mroz200w,1);
Y = mroz200w(:,1);
X = [ones(N,1) mroz200w(:,2:end)];
X(:,5) = X(:,5)/100;
k = size( X, 2 );
nloop = 5500; 
burnin = 500;

% parameters for the jumping distribution
% ues MLE as the location and scale parameters
[muj,fval,exitflag,output,grad,hessian] = fminunc(@(x)-lblogit(x,Y,X),zeros(8, 1)); 
Vj = hessian\speye(k);
nuj = 10;
CVj = chol(Vj)';

% parameters for the prior
beta0 = sparse(k,1);
Vbeta0 = 10*speye(k);

% set initial values
store_beta = zeros(nloop-burnin,k);
beta = zeros(k,1);
start_time = clock;

    
% MCMC starts here
for loop = 1 : nloop
    betac = muj + CVj*randn(k,1)/sqrt(gamrnd(nuj/2,2/nuj));
    % evaluate the log posterior ratio
    lpr = lmvnpdf(betac,beta0,Vbeta0) + lblogit(betac,Y,X) ...
        - lmvnpdf(beta,beta0,Vbeta0) - lblogit(beta,Y,X);
    alpha = lpr + lmtpdf(beta,nuj,muj,Vj) - lmtpdf(betac,nuj,muj,Vj);     
    if (exp(alpha)>rand)
        beta = betac;
    end    
    if loop > burnin
        store_beta(loop-burnin,:) = beta';
    end    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

%% compute ML using CE
ndraws = 5000;
nbigloop = 10;
logit_CE
ML = mean(sml)
MLstd = std(sml)/sqrt(nbigloop)
