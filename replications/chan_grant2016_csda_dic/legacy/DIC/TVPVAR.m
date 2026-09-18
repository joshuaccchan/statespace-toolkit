% This is the main run file for estimating the TVP-VAR(1) model and 
% computing the observed-data DIC (DIC2), complete-data DIC (DIC5), 
% and conditional DIC (DIC7)
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
DIC_choice = 2;   % choose between DIC2, DIC5 or DIC7 (2, 5, 7, respectively)
R = 10;           % number of parallel chains
nloop = 11000;    % number of simulations for each chain
burnin = 1000;    % number of initial draws discarded
if ~ismember(DIC_choice, [2 5 7])
    warning('DIC_choice should be 2, 5 or 7');
end
load USdata.csv;  % [GDP growth, TBill rate, unemployment rate, CPI rate]
Y0 = USdata(1:3,:);
shortY = USdata(4:end,:);
[T n] = size(shortY);
Y = reshape(shortY',T*n,1);
p = 1;        % number of AR lags
k = n^2*p+n;  % dim of states
Tk = T*k; Tn = T*n; k2 = k^2; nnp1 = n*(n+1);

    %% initialize for storage
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);
    %% prior
nuSig = n+3;  SSig0 = eye(n);
nuOmega = 5;  SOmega0 = (nuOmega-1)*.005*ones(k,1);
b0 = sparse(k,1); Q0 = 5*ones(k,1); % store only the diag elements
prior = @(S,O) -(nuSig+n+1)/2*log(det(S)) - .5*trace(S\SSig0) + ...
        sum(-(nuOmega+1)*log(O) - SOmega0./O);
    
%% construct and compute a few thigns
X = zeros(T,n*p);
for i=1:p
    X(:,(i-1)*n+1:i*n) = [Y0(3-i+1:end,:); shortY(1:T-i,:)];
end
bigX = SURform([ones(n*T,1) kron(X,ones(n,1))]);
H = speye(Tk,Tk) - sparse(k+1:Tk,1:(T-1)*k,ones(1,(T-1)*k),Tk,Tk);
newnuSig = nuSig + T;
newnuOmega = nuOmega + (T-1)/2;
alp = H\[b0;sparse((T-1)*k,1)];

disp('Starting MCMC.... ');
start_time = clock;
for bigloop = 1:R    
    disp(' ' );
    disp(  [ num2str( R-bigloop+1 ) ' more chains to go... ' ] );    
        %% initialize the Markov chain
    Sig = cov(USdata);
    Omega = .01*ones(k,1); % store only the diagonal elements
    invSig = Sig\speye(n);
    invOmega = 1./Omega;
    beta = zeros(Tk,1); 
    for loop = 1:nloop    
            %% sample beta    
        XinvSig = bigX'*kron(speye(T),invSig);
        XinvSigX = XinvSig*bigX;
        HinvSH = H'*sparse(1:Tk,1:Tk,[1./Q0; repmat(invOmega,T-1,1)]')*H;
        Kbeta = HinvSH + XinvSigX;    
        C = chol(Kbeta,'lower');                
        betahat = C'\(C\(HinvSH*alp + XinvSig*Y));
        beta = betahat + C'\randn(Tk,1);
            %% sample Sig
        e1 = reshape(Y-bigX*beta,n,T);
        newS1 = SSig0 + e1*e1';
        invSig = wishrnd(newS1\speye(n),newnuSig);
        Sig = invSig\speye(n);
            %% sample Omega
        e2 = reshape(H*beta,k,T);
        newS2 = SOmega0 + sum(e2(:,2:end).^2,2)/2;
        invOmega = gamrnd(newnuOmega,1./newS2);
        Omega = 1./invOmega;  
        if loop>burnin
            i=loop-burnin;            
            if DIC_choice == 2     % integrated likelihood DIC
                llike = intlike_tvpvar(Y,bigX,Sig,Omega,Q0,b0);
            elseif DIC_choice == 5 % complete-data likelihood DIC
                llike = like_tvpvar(Y,bigX,beta,Sig,Omega,Q0,b0); 
            elseif DIC_choice == 7 % conditional likelihood DIC
                llike = conlike_tvpvar(Y,bigX,beta,Sig);    
            end    
            if DIC_choice == 2 || DIC_choice == 5
                lpost = llike + prior(Sig,Omega);
            elseif DIC_choice == 7
                lpost = like_tvpvar(Y,bigX,beta,Sig,Omega,Q0,b0) + ...
                    + prior(Sig,Omega);
            end                
            store_llike(i) = llike; 
            store_lpost(i) = lpost;
        end    
        if ( mod( loop, 5000 ) ==0 )
            disp(  [ num2str( loop ) ' loops... ' ] )
        end    
    end

    %% compute DIC
    [~, id] = max(store_lpost);
    DIC = -4*mean(store_llike) + 2*store_llike(id);
    store_DIC(bigloop) = DIC;    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

DIC = mean(store_DIC);             % DIC
DICNSE = std(store_DIC)/sqrt(R);   % numerical standard error
