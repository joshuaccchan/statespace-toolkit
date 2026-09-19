%% GOLDEN-RUN VARIANT PATCH (2026-09-19): identical to legacy/SV/plainSV.m except that
%% a numeric capture is appended at the end, which prints the results the script computes
%% and saves their posterior means and 5% and 95% quantiles to golden_capture.mat.
%% The computation is untouched.
% % =======================================================================
% % Stochastic volatility model with a constant mean
% %
% % y_t = mu + epsilon_t,                    epsilon_t ~ N(0,exp(h_t)),
% % h_t = muh + phih(h_{t-1}-muh) + zeta_t,  zeta_t ~ N(0,sigh2),
% % 
% % See Chan, J.C.C. and Hsiao, C.Y.L (2014). Estimation of Stochastic
% % Volatility Models with Heavy Tails and Serial Dependence. 
% % In: I. Jeliazkov and X.S. Yang (Eds.), Bayesian Inference in the 
% % Social Sciences, 159-180, John Wiley & Sons, New York.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
nloop = 21000;
burnin = 1000;
load 'AUDUSD.csv';
y = AUDUSD; 
T = length(y);

%% prior
invVmu = 1/5;
phih0 = .95; invVphih = 1;
muh0 = 0; invVmuh = 1/5;
nuh = 10; Sh = .02*(nuh-1);

disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;    
    
% initialize the Markov chain
sigh2 = .05;
phih = .95;
muh = 1;
mu = mean(y);
h = log(var(y)*.8)*ones(T,1);

% initialize for storage
store_theta = zeros(nloop - burnin,4); % [mu muh phih sigh2]
store_exph = zeros(nloop - burnin,T);  % store exp(h_t/2)

%% compute a few things outside the loop
newnuh = T/2 + nuh;
rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

for loop = 1:nloop
        %% sample mu    
    invexph = exp(-h);
    Dmu = 1/(invVmu + sum(invexph));
    muhat = Dmu*sum(invexph.*y);
    mu = muhat + sqrt(Dmu)*randn;        
        %% sample h, muh, phih, sigh2
    Ystar = log((y-mu).^2 + .0001);
    [h muh phih sigh2] = SV(Ystar,h,muh,phih,sigh2,[muh0 invVmuh ...
        phih0 invVphih nuh Sh]);    
    if ( mod( loop, 2000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
    if loop>burnin
        i = loop-burnin;
        store_exph(i,:) = exp(h/2)'; 
        store_theta(i,:) = [mu muh phih sigh2];
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

thetahat = mean(store_theta)';
exphhat = mean(store_exph)'; 
exphlb = quantile(store_exph,.05)';
exphub = quantile(store_exph,.95)';
figure;
tid = linspace(2005,2013,T)';
figure; plot(tid, [exphhat exphlb exphub]);
box off; xlim([2005 2013]);



%% ===== GOLDEN CAPTURE, appended by the savegolden patch; the computation above is untouched =====
golden_named = {};
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
