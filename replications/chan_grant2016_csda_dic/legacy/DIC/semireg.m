% This is the main run file for estimating a semiparametric regression
% and the observed-data DIC (DIC2), complete-data DIC (DIC5) and conditional
% DIC (DIC7) under the model
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
%
% The BMI data are restricted access and so they cannot be uploaded
% directly. The data we use in this paper, however, are the same
% as the data that were used in the paper, “The Wages of BMI: Bayesian 
% Analysis of a Skewed Treatment-Response Model with Nonparametric 
% Endogeneity” (Kline and Tobias, 2008).

clear; clc;
DIC_choice = 2;   % choose between DIC2, DIC5 or DIC7 (2, 5, 7, respectively)
R = 10;           % number of parallel chains
nloop = 11000;    % number of simulations for each chain
burnin = 1000;    % number of initial draws discarded
if ~ismember(DIC_choice, [2 5 7])
    warning('DIC_choice should be 2, 5 or 7');
end

%% simulate data 
n = 1000;
m = n;            % number of distinct functional values 
                  % (in our example all values are distinct)
k = 4;            % number of regressors in X
sig2 = .02;
beta = 1 + randn(k,1);  % no intercepts!
Z = -2 + 4*rand(n,1);   % draw from U(-2,2)
X = 2*rand(n,k);        % draw from U(0,2)
g = @(z) .15*z + .3*exp(-4*(z+1).^2) + .7*exp(-16*(z-1).^2);
y = X*beta + g(Z) + sqrt(sig2)*randn(n,1);
[Z id] = sort(Z);       % sort Z in ascending order
y = y(id);
X = X(id,:);
D = speye(m);           % selection matrix 
                        % (all values in Z are distinct in the simulated data)

    %% prior
beta0 = sparse(k,1); invVbeta = speye(k);
nusig2 = 3; Ssig2 = 1*(nusig2-1);       % IG prior for sig2
nutau = 3; Stau = 5*10^(-6)*(nutau-1);  % IG prior for tau
V1 = 100;
V2 = 100;
prior = @(b,S,t) - .5*b'*invVbeta*b + sum(-(nusig2+1)*log(S) - Ssig2./S) + ...
        sum(-(nutau+1)*log(t) - Stau./t);

    %% compute a few things
XX = X'*X;
DD = D'*D;
del = (Z(2:end) - Z(1:end-1))';
G1 = sparse(1:m,1:m,[1 1 1./del(2:end)],m,m);
G2 = sparse(3:m,2:m-1, -(1./del(2:end)+1./del(1:end-1)),m,m);
G3 = sparse(3:m,1:m-2,1./del(1:end-1),m,m);
G = G1 + G2 + G3;

    %% initialize for storage
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);

disp('Starting MCMC.... ');
start_time = clock;

for bigloop = 1:R;
    
    disp(' ' );
    disp(  [ num2str( R-bigloop+1 ) ' more chains to go... ' ] );  

        %% initialize the Markov chain
    beta = XX\(X'*y);
    sig2 = (y-X*beta)'*(y-X*beta)/n;
    tau = Stau/(nutau-1);

    for loop=1:nloop
            %% sample theta
        invOmega = sparse(1:m,1:m,[1/V1 1/V2 1/tau*ones(1,m-2)]);
        GinvOmegaG = G'*invOmega*G;
        invDtheta = DD/sig2 + GinvOmegaG;
        thetahat = invDtheta\(D'*(y-X*beta)/sig2);
        theta = thetahat + chol(invDtheta,'lower')'\randn(m,1); 
    
            %% sample beta
        invDbeta = invVbeta + XX/sig2;
        betahat = invDbeta\(invVbeta*beta0 + X'*(y-D*theta)/sig2);
        beta = betahat + chol(invDbeta,'lower')'\randn(k,1);
    
            %% sample sig2
        e = y-D*theta-X*beta;
        sig2 = 1/gamrnd(nusig2+n/2, 1/(Ssig2+e'*e/2));
    
            %% sample tau
        Gtheta = G*theta;
        e = Gtheta(3:end);
        tau = 1/gamrnd(nutau+(m-2)/2, 1/(Stau+e'*e/2));

        if loop>burnin
            i = loop-burnin;
            ymXbeta = y-X*beta;
            if  DIC_choice == 2     % integrated likelihood
                llike = intlike_semireg(ymXbeta,G,D,sig2,tau,V1,V2);
            elseif DIC_choice == 5; % complete-data likelihood
                llike = like_semireg(ymXbeta,G,D,theta,sig2,tau,V1,V2);
            elseif DIC_choice == 7; % conditional likelihood
                llike = conlike_semireg(ymXbeta,D,theta,sig2);
            end
            if DIC_choice == 2 || DIC_choice == 5;
                lpost = llike + prior(beta,sig2,tau);
            elseif DIC_choice == 7;
                lpost = like_semireg(ymXbeta,G,D,theta,sig2,tau,V1,V2) ...
                    + prior(beta,sig2,tau);
            end
            store_llike(i) = llike;
            store_lpost(i) = lpost;
        end
    
        if ( mod( loop, 10000 ) ==0 )
            disp(  [ num2str( loop ) ' loops... ' ] )
        end
    end
        %% compute DIC
    [~,id] = max(store_lpost);
    DIC = -4*mean(store_llike) + 2*store_llike(id);
    store_DIC(bigloop) = DIC;

end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

DIC = mean(store_DIC);             % (observed-data) DIC
DICNSE = std(store_DIC)/sqrt(R);   % numerical standard error  

