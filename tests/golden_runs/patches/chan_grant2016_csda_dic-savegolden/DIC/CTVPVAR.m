%% GOLDEN-RUN VARIANT PATCH (2026-09-19): identical to legacy/DIC/CTVPVAR.m except that
%% a numeric capture is appended at the end, which prints the results the script computes
%% and saves their posterior means and 5% and 95% quantiles to golden_capture.mat.
%% The computation is untouched.
% This is the main run file for estimating a variant of the TVP-VAR where
% the first equation has constant coefficients. This file also computes
% the observed-data DIC (DIC2), complete-data DIC (DIC5), and conditional
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

clear; clc;
DIC_choice = 2;   % choose between DIC2, DIC5 or DIC7 (2, 5, 7, respectively)
R = 10;           % number of parallel chains
nloop = 11000;    % number of simulations for each chain
burnin = 1000;    % number of initial draws discarded
eq_const = 3;     % select the equation that has constant coefficients
                  % 1: GDP growth; 2: TBill rate; 3: unemployment rate; 
                  % 4: CPI inflation rate
if ~ismember(DIC_choice, [2 5 7])
    warning('DIC_choice should be 2, 5 or 7');
end
eq_tv = setxor(eq_const, 1:4);
load USdata.csv; 
USdata = USdata(:, [eq_const eq_tv]); 
Y0 = USdata(1:3,:);
shortY = USdata(4:end,:);
[T n] = size(shortY);
Y = reshape(shortY',T*n,1);
p = 1;        % number of AR lags
k = 1+n*p;    % number of constant coefficients
q = (n-1)*n*p+(n-1); % dim of states
Tq = T*q; Tn = T*n;
    %% initialize for storage
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);
    %% prior
nuSig = n+3;  SSig0 = eye(n);
nuOmega = 5;  SOmega0 = (nuOmega-1)*.005*ones(q,1);
b0 = sparse(q,1); Q0 = 5*ones(q,1); % store only the diag elements
gam0 = zeros(n+1,1); invVgam = speye(n+1)/5;
prior = @(g,S,O) -.5*(g-gam0)'*invVgam*(g-gam0) + ...
    -(nuSig+n+1)/2*log(det(S)) - .5*trace(S\SSig0) + ...
    +sum(-(nuOmega+1)*log(O) - SOmega0./O);    

%% construct and compute a few thigns
H = speye(Tq,Tq) - sparse(q+1:Tq,1:(T-1)*q,ones(1,(T-1)*q),Tq,Tq);
w1 = reshape([ones(T,1) USdata(3:end-1,:)]',T*(n+1),1);
r1 = kron(1:n:(T-1)*n+1,ones(1,n+1))';
c1 = kron(ones(1,T),(1:n+1))';
Z = sparse(r1,c1,w1,Tn,k);
tempid = reshape(1:Tn,n,T)';
tempid = reshape(tempid(:,2:end)',T*(n-1),1);
r2 = kron(tempid,ones(n+1,1));
c2 = (1:T*(n-1)*(n+1))';
w2 = reshape([ones((n-1)*T,1) kron(USdata(3:end-1,:),ones(n-1,1))]',T*(n-1)*(n+1),1);
X = sparse(r2,c2,w2);
newnuSig = nuSig + T;
newnuOmega = nuOmega + (T-1)/2;
alp = H\[b0;sparse((T-1)*q,1)];
    
disp('Starting MCMC.... ');
start_time = clock;
for bigloop = 1:R
    disp(' ' );
    disp(  [ num2str( R-bigloop+1 ) ' more chains to go... ' ] );    
    
        %% initialize the Markov chain
    Sig = cov(USdata);
    Omega = .01*ones(q,1); % store only the diag elements
    invSig = Sig\speye(n);
    invOmega = 1./Omega;
    beta = .01 * randn(Tq,1);
    gam = .01 * randn(k,1);    
    for loop = 1:nloop    
          %% sample gam
        ZinvSig = Z'*kron(speye(T),invSig);    
        invDgam = ZinvSig*Z + invVgam;
        Cgam = chol(invDgam,'lower');
        gamhat = invDgam\(ZinvSig*(Y-X*beta) + invVgam*gam0);
        gam = gamhat + Cgam'\randn(n+1,1);    
            %% sample beta    
        XinvSig = X'*kron(speye(T),invSig);
        XinvSigX = XinvSig*X;
        HinvSH = H'*sparse(1:Tq,1:Tq,[1./Q0; repmat(invOmega,T-1,1)]')*H;
        Kbeta = HinvSH + XinvSigX;    
        C = chol(Kbeta,'lower');                
        betahat = C'\(C\(HinvSH*alp + XinvSig*(Y - Z*gam)));
        beta = betahat + C'\randn(Tq,1);
            %% sample Sig
        e1 = reshape(Y-X*beta-Z*gam,n,T);
        newS1 = SSig0 + e1*e1';
        invSig = wishrnd(newS1\speye(n),newnuSig);
        Sig = invSig\speye(n);
            %% sample Omega
        e2 = reshape(H*beta,q,T);
        newS2 = SOmega0 + sum(e2(:,2:end).^2,2)/2;
        invOmega = gamrnd(newnuOmega,1./newS2);
        Omega = 1./invOmega;  
        if loop>burnin
            i=loop-burnin;        
            if DIC_choice == 2     % integrated likelihood
                llike = intlike_tvpvar(Y-Z*gam,X,Sig,Omega,Q0,b0);                        
            elseif DIC_choice == 5 % complete-data likelihood 
                llike = like_tvpvar(Y-Z*gam,X,beta,Sig,Omega,Q0,b0);                         
            elseif DIC_choice == 7 % conditional likelihood
                llike = conlike_tvpvar(Y-Z*gam,X,beta,Sig);                        
            end    
            if DIC_choice == 2 || DIC_choice == 5
                lpost = llike + prior(gam,Sig,Omega);
            elseif DIC_choice == 7
                lpost = like_tvpvar(Y-Z*gam,X,beta,Sig,Omega,Q0,b0) + ...
                    + prior(gam,Sig,Omega);
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

%% ===== GOLDEN CAPTURE, appended by the savegolden patch; the computation above is untouched =====
golden_named = {'DIC_choice', 'DIC', 'DICNSE'};
golden_draws = [who('store_*'); {}'];
golden = struct('matlab', version);
for golden_v = golden_named(:)'
    if exist(golden_v{1}, 'var')
        golden.(golden_v{1}) = eval(golden_v{1});
        fprintf('%s = %s\n', golden_v{1}, mat2str(golden.(golden_v{1}), 8));
    end
end
for golden_v = golden_draws(:)'
    if exist(golden_v{1}, 'var')
        golden_x = eval(golden_v{1});
        if isnumeric(golden_x) && ismatrix(golden_x) && size(golden_x,1) > 1
            golden.([golden_v{1} '_mean']) = mean(golden_x, 1);
            golden.([golden_v{1} '_q05']) = quantile(golden_x, .05, 1);
            golden.([golden_v{1} '_q95']) = quantile(golden_x, .95, 1);
            if size(golden_x, 2) <= 12
                fprintf('%s posterior means: %s\n', golden_v{1}, mat2str(mean(golden_x, 1), 6));
            end
        end
    end
end
save('golden_capture.mat', 'golden');
