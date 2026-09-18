function Post = sy_SW_original_v3(Data, Prior, Ini, MCMC, varargin)

    flag.plkl = false;

    if isempty(varargin) % infer posterior
        y = Data.y;
        T = Data.T;
        ystar = Ini.ystar;
        g = Ini.g;
        h = Ini.h;
    elseif (strcmp(varargin{1}, 'plkl')) %calculate predictive likelihood
        T = Data.T;
        y = Data.y(1:T);
        yc = Data.y(T+1);
        ystar = Ini.ystar(1:T);
        g = Ini.g(1:T);
        h = Ini.h(1:T);
        
        flag.plkl = true;
    else
        fprintf('not a valid input');
        return 
    end
    %% Release variables
    sigma2h = Ini.sigma2h; 
    sigma2g = Ini.sigma2g; 
    
    vh = Prior.vh;
    vg = Prior.vg;
    sh = Prior.sh;
    sg = Prior.sg;
    Vystar = Prior.Vystar;  %varaince of y1* is Vystar * exp(g1); variance of y1 is var(y1*) + exp(h1)
    mystar = Prior.mystar; % mean of y1star
    Vg = Prior.Vg; % variance of g1
    mg = Prior.mg; % mean of g1
    Vh = Prior.Vh; % variance of h1
    mh = Prior.mh; % mean of h1
    
    Burnin = MCMC.Burnin;
    Nuse = MCMC.Nuse;
    Nsim = Burnin + Nuse;
    
    %% Output variable
    if ~flag.plkl 
        Post.sigma2h = zeros(Nuse, 1);
        Post.sigma2g = zeros(Nuse, 1);
        Post.g1 = zeros(Nuse, 1);
        Post.h1 = zeros(Nuse, 1);
        Post.ystar = zeros(T, Nuse); 
        Post.expg = zeros(T, Nuse);

        Post.ystar_pm = zeros(T, 1);
        Post.expg_pm = zeros(T, 1);
        Post.exph_pm = zeros(T, 1);
    end
    if flag.plkl
        Post.plkl = zeros(Nuse, 1);
    end

    %% Kim, Shephard and Chib(1998) weights, means and variances
    KSC_w = [0.00730; 0.10556; 0.00002; 0.04395; 0.34001; 0.24566; 0.25750];
    KSC_m = [-10.12999; -3.97281; -8.56686; 2.77786; 0.61942; 1.79518; -1.08819] - 1.2704;
    KSC_v2 = [5.79596; 2.61369; 5.17950; 0.16735; 0.64009; 0.34023; 1.26261];

    KSC_W = repmat(KSC_w', T, 1);
    KSC_M = repmat(KSC_m', T, 1);
    KSC_V = repmat(sqrt(KSC_v2)', T, 1);
    
    %% some variables
    H = speye(T) - spdiags(ones(T-1,1),-1,T,T);

    %% MCMC
    for iter = 1 : Nsim
    
        if mod(iter, 5000) == 0
            fprintf('iteration: %d\n', iter);
        end
        
        % Sg    
        ytilde = [log((ystar(1)-mystar)^2/Vystar); log(diff(ystar).^2)];
        PS = cumsum(KSC_W .* normpdf(repmat(ytilde-g, 1, 7), KSC_M, KSC_V),2);
        S =  1 + sum(repmat(rand(T,1) .* PS(:, 7), 1, 7 ) > PS, 2);

        % g
        ds = KSC_m(S);
        Omegas_inv = spdiags(1./KSC_v2(S), 0, T, T);
        Sv_inv = spdiags([1/Vg, repmat(1/sigma2g, 1, T-1)]', 0, T, T);
        Kg = Omegas_inv + H' * Sv_inv * H;
        ghat = Kg\(Omegas_inv * (ytilde-ds) + [mg/Vg; zeros(T-1, 1)]);
        P = chol(Kg);
        g = ghat + P\randn(T, 1);
    
        % Z or Sh
        ytilde = log(max((y-ystar).^2, eps));
        PS = cumsum(KSC_W .* normpdf(repmat(ytilde-h, 1, 7), KSC_M, KSC_V),2);
        Z =  1 + sum(repmat(rand(T,1) .* PS(:, 7), 1, 7 ) > PS, 2);
        
        % h
        dz = KSC_m(Z);
        Omegaz_inv = spdiags(1./KSC_v2(Z), 0, T, T);
        Sv_inv = spdiags([1/Vh, repmat(1/sigma2h, 1, T-1)]', 0, T, T);
        Kh = Omegaz_inv + H'*Sv_inv*H;
        hhat = Kh\(Omegaz_inv * (ytilde - dz) + [mh/Vh; zeros(T-1, 1)]);
        P = chol(Kh);
        h = hhat + P\randn(T, 1);
        
        % ystar
        Systar_inv = spdiags([exp(-g(1))/Vystar; exp(-g(2:T))], 0, T, T);
        Omegah_inv = spdiags(exp(-h), 0, T, T);
        Kystar = H' * Systar_inv * H + Omegah_inv;
        ystarhat = Kystar\ ([mystar * exp(-g(1)) / Vystar; zeros(T-1, 1)] + Omegah_inv * y);
        P = chol(Kystar);
        ystar = ystarhat + P\randn(T, 1);
    
        %sigma2h
        shhat = sh + sum(diff(h).^2);
        vhhat = vh + T - 1;
    
        sigma2h = 1/gamrnd(vhhat/2, 2/shhat);
        
        %sigma2g
        sghat = sg + sum(diff(g).^2);
        vghat = vg + T - 1;
    
        sigma2g = 1/gamrnd(vghat/2, 2/sghat);

    
        % save posterior
        if iter > Burnin
            ii = iter - Burnin;
            if ~flag.plkl
                Post.sigma2h(ii) = sigma2h;
                Post.sigma2g(ii) = sigma2g;
                Post.g1(ii) = g(1);
                Post.h1(ii) = h(1);
                
                Post.expg(:, ii) = exp(g/2); %std
                Post.ystar(:, ii) = ystar; 

                Post.ystar_pm= Post.ystar_pm + ystar;
                Post.expg_pm= Post.expg_pm + exp(g/2); %std
                Post.exph_pm= Post.exph_pm + exp(h/2); %std
            end
            
            
          %% predictive likelihood
            if flag.plkl
                % simulate g forward
                gc = g(T) + randn * sqrt(sigma2g);
                % simulate h forward 
                hc = h(T) + randn * sqrt(sigma2h);
                % predictive likelihood
                Post.plkl(ii) = normpdf(yc, ystar(T), sqrt(exp(gc)+exp(hc)));
            end
        end
    end
    
    if ~flag.plkl
        Post.ystar_pm = Post.ystar_pm / Nuse;
        Post.expg_pm = Post.expg_pm / Nuse;
        Post.exph_pm = Post.exph_pm / Nuse;    
    
        Post.last.sigma2h = sigma2h; % h and g equation (y and ystar log volatility): error variance
        Post.last.ystar = ystar; % ystar values
        Post.last.g = g; % h equation (y log volatility)
        Post.last.h = h; % g equation (ystar log volatility)
    end
end