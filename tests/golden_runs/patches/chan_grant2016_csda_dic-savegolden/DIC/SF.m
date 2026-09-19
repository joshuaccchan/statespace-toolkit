%% GOLDEN-RUN VARIANT PATCH (2026-09-19): identical to legacy/DIC/SF.m except that
%% a numeric capture is appended at the end, which prints the results the script computes
%% and saves their posterior means and 5% and 95% quantiles to golden_capture.mat.
%% The computation is untouched.
% This is the main run file for estimating a static factor model and 
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
DIC_choice = 5;   % choose between DIC2, DIC5 or DIC7 (2, 5, 7, respectively)
R = 10;           % number of parallel chains
q = 4;            % number of factors (from 1 to 4)
nloop = 11000;    % number of simulations for each chain
burnin = 1000;    % number of initial draws discarded
if ~ismember(DIC_choice, [2 5 7])
    warning('DIC_choice should be 2, 5 or 7');
end
load 'returns_1_10.csv'
load 'rfcsrp.csv'
shortY = (returns_1_10 - repmat(rfcsrp,1,10))*100;
[T n] = size(shortY);
shortY = shortY(:,[6 10 1 2 3 4 5 7 8 9]);
m = 0;    % number of variables in z
p = 1;    % number of lags for z-1, e.g. p = 1 => no lags 
k = m*p+1;
nk = n*k;
if q==1
    subA = [(2:n)' ones(n-1,1)]; % free elements in A  
    subID = [1 1];               % elements in A for that are set to 1
elseif q==2
    subA = [2 1; kron((3:n)',ones(2,1)) repmat((1:2)',n-2,1)];
    subID = [1 1; 2 2];
elseif q==3
    subA = [2 1; 3 1; 3 2; kron((4:n)',ones(3,1)) repmat((1:3)',n-3,1)];
    subID = [1 1; 2 2; 3 3];    
elseif q ==4
    subA = [2 1; 3 1; 3 2; 4 1; 4 2; 4 3; ...
        kron((5:n)',ones(4,1)) repmat((1:4)',n-4,1)];
    subID = [1 1; 2 2; 3 3; 4 4];        
end
indA = sub2ind([n q], subA(:,1),subA(:,2));
indID = sub2ind([n q], subID(:,1), subID(:,2));    
nA = length(indA);    
Y = reshape(shortY',T*n,1);       
X = ones(T,1);
%% initialize for storage
Tn = T*n;
Tq = T*q;
nm1 = n-1;
qm1 = q-1;
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);

%% prior
invVa = 1;
invVb = 1;
nuSig = 3; SSig = 1*(nuSig-1)*ones(n,1);
nuOmega = 3; SOmega = 1*(nuOmega-1)*ones(q,1);
prior = @(a,b,S1,S2) -.5*a'*invVa*a - .5*b'*invVb*b + ...
        sum(-(nuSig+1)*log(S1) - SSig./S1) + sum(-(nuOmega+1)*log(S2) - SOmega./S2);

%% compute a few things outside the loop
newnuSig = nuSig + T/2;
newnuOmega = nuOmega + T/2;
    
disp('Starting MCMC.... ');
start_time = clock;
for bigloop = 1:R;
    disp(' ' );
    disp(  [ num2str( R-bigloop+1 ) ' more chains to go... ' ] );    
    
    %% initialize the Markov chain
    Sig = ones(n,1);
    invSig = 1./Sig;
    invOmega = ones(q,1);
    B = ((X'*X)\(X'*shortY))';
    A = zeros(n,q);
    A(indA) = .2*rand(nA,1);
    A(indID) = 1;
    a = A(indA); 
    YmXb = reshape( (shortY - X*B')',Tn,1);

    for loop = 1:nloop 
    
        %% sample f   
        F0 = sparse(1:Tq,1:Tq,repmat(invOmega,T,1)');
        spA = sparse(A);    
        AinvSig1 = (spA.*repmat(invSig,1,q))';
        invDf = F0 + kron(speye(T), AinvSig1*spA); 
        df = kron(speye(T), AinvSig1)*YmXb;
        Cf = chol(invDf);
        fhat = Cf\(Cf'\df);
        f = fhat + Cf\randn(Tq,1);   
        shortf = reshape(f,q,T)';
    
        %% sample A and beta (b) by rows
        counta = 0;
        for i=1:n
            idAi = subA(subA(:,1)==i,2);
            nai = length(idAi);
            ki = nai + k;
            bigX = [X shortf(:,idAi)]; 
            invDthe = spdiags([invVb*ones(k,1); ...
                invVa*ones(nai,1)],0,ki,ki) + bigX'*bigX*invSig(i);
            if i<=q
                dthe = bigX'*(shortY(:,i)-shortf(:,i))*invSig(i);
            else
                dthe = bigX'*shortY(:,i)*invSig(i);
            end
            Cthe = chol(invDthe);
            thehat = Cthe\(Cthe'\dthe);
            theta = thehat + Cthe\randn(ki,1);
            B(i,:) = theta(1:k)';
            if ~isempty(idAi)
                A(i,idAi) = theta(k+1:end)'; 
            end
            counta = counta + nai;
        end
        b = reshape(B',nk,1);
        a = A(indA);
        YmXb = reshape((shortY-X*B')',Tn,1);    
 
        %% sample Sig
        err = reshape(YmXb,n,T)-A*shortf';
        newSSig = SSig + sum(err.^2,2)/2;
        invSig = gamrnd(newnuSig, 1./newSSig);
        Sig = 1./invSig;
    
        %% sample Omega
        newSOmega = SOmega +  sum(shortf.^2)'/2;
        invOmega = gamrnd(newnuOmega, 1./newSOmega);
        Omega = 1./invOmega;
    
        if loop>burnin
            i = loop-burnin;   
            if  DIC_choice == 2     % integrated likelihood
                llike = intlike_sf(shortY,YmXb,A,invSig,invOmega);
            elseif DIC_choice == 5; % complete-data likelihood
                llike = like_sf(shortY,YmXb,A,shortf,invSig,invOmega);
            elseif DIC_choice == 7; % conditional likelihood
                llike = conlike_sf(shortY,YmXb,A,shortf,invSig);
            end
            if DIC_choice == 2 || DIC_choice == 5;
                lpost = llike + prior(a,b,Sig,Omega);
            elseif DIC_choice == 7;
                lpost = like_sf(shortY,YmXb,A,shortf,invSig,invOmega) + prior(a,b,Sig,Omega);
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
