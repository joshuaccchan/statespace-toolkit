function Post = sy_CS2(Data, Prior, Ini, MCMC)


    %% data
    y = Data.y; % inflation
    x = Data.x; % mean measure
    z = Data.z; % RV measure
    T = Data.T;
    logz = log(z);

    %% initial values
    sigma2h = Ini.sigma2h; % h equation (y log volatility): error variance
    sigma2g = Ini.sigma2g; % g equation (ystar log volatility): error variance
    sigma2z = Ini.sigma2z; % z equation (RV measurement): error variance
    sigma2x = Ini.sigma2x; % x equation (ystar measurement): error variance
    ystar = Ini.ystar; % ystar values
    g = Ini.g; % h equation (y log volatility)
    h = Ini.h; % g equation (ystar log volatility)
    a = Ini.a; % z equation (RV measurement): coefficients
    b = Ini.b; % x equation (ystar measurement): coefficients


    %% prior parameters
    vh = Prior.vh;% h equation (y log volatility): error variance
    sh = Prior.sh;% h equation (y log volatility): error variance
    vg = Prior.vg; % g equation (ystar log volatility): error variance
    sg = Prior.sg; % g equation (ystar log volatility): error variance
    vz = Prior.vz; % z equation (RV measurement): error variance
    sz = Prior.sz; % z equation (RV measurement): error variance
    vx = Prior.vx; % x equation (ystar measurement): error variance
    sx = Prior.sx; % x equation (ystar measurement): error variance
    Vystar = Prior.Vystar; %varaince of ystar1 is Vystar * exp(g1); variance of y1 is var(y1*) + exp(h1)
    mystar = Prior.mystar; %mean of ystar1
    Vg = Prior.Vg; % variance of g1
    mg = Prior.mg; % mean of g1
    Vh = Prior.Vh; % variance of h1
    mh = Prior.mh; % mean of h1
    ma = Prior.ma; % mean of coefficients of the z equation
    Ka = Prior.Ka; % Precision of coefficients of the z equation
    mb = Prior.mb; % mean of coefficients of the x equation
    Kb = Prior.Kb; % Precision of coefficients of the x equation

    %% MCMC setting
    Burnin = MCMC.Burnin;
    Nuse = MCMC.Nuse;
    Nsim = Burnin + Nuse;
    
    %% Output variable
    Post.sigma2h = zeros(Nuse, 1); %h
    Post.sigma2g = zeros(Nuse, 1); %g
    Post.sigma2z = zeros(Nuse, 1); %z
    Post.sigma2x = zeros(Nuse, 1); %x
    Post.a = zeros(Nuse, 2);
    Post.b = zeros(Nuse, 2);
    Post.g1 = zeros(Nuse, 1);
    Post.h1 = zeros(Nuse, 1);
    Post.ystar = zeros(T, Nuse); 
    Post.ystar_pm = zeros(T, 1);
    Post.expg_pm = zeros(T, 1);
    Post.g_pm = zeros(T, 1);
    Post.exph_pm = zeros(T, 1);
    Post.h_pm = zeros(T, 1);
    %Post.expg = zeros(T, Nuse);

    %% Kim, Shephard and Chib(1998) weights, means and variances
    KSC_w = [0.00730; 0.10556; 0.00002; 0.04395; 0.34001; 0.24566; 0.25750];
    KSC_m = [-10.12999; -3.97281; -8.56686; 2.77786; 0.61942; 1.79518; -1.08819] - 1.2704;
    KSC_v2 = [5.79596; 2.61369; 5.17950; 0.16735; 0.64009; 0.34023; 1.26261];

    KSC_W = repmat(KSC_w', T, 1);
    KSC_M = repmat(KSC_m', T, 1);
    KSC_V = repmat(sqrt(KSC_v2)', T, 1);
    
    %% some variables
    H = speye(T) - spdiags(ones(T-1,1),-1,T,T);
    IT = speye(T);
    X = ones(T, 2);

    %% MCMC
    for iter = 1 : Nsim
    
        if mod(iter, 5000) == 0
            fprintf('iteration: %d\n', iter);
        end
        
        % Sg
        ytilde = [log((ystar(1)-mystar)^2/Vystar); log(diff(ystar).^2)];
        ytilde = max(ytilde, log(eps)); %safeguard
        PS = cumsum(KSC_W .* normpdf(repmat(ytilde-g, 1, 7), KSC_M, KSC_V),2);
        Sg =  1 + sum(repmat(rand(T,1) .* PS(:, 7), 1, 7 ) > PS, 2);

        % g
        ds = KSC_m(Sg);
        Omegas_inv = spdiags(1./KSC_v2(Sg), 0, T, T);
        Sv_inv = spdiags([1/Vg, repmat(1/sigma2g, 1, T-1)]', 0, T, T);
        Kg = Omegas_inv + H' * Sv_inv * H + a(2)^2 / sigma2z * IT;
        ghat = Kg\(Omegas_inv * (ytilde-ds) + [mg/Vg; zeros(T-1, 1)] + a(2)/sigma2z * (logz - a(1)));
        P = chol(Kg);
        g = ghat + P\randn(T, 1);
    
        % Sh
        ytilde = log(max((y-ystar).^2, eps));
        PS = cumsum(KSC_W .* normpdf(repmat(ytilde-h, 1, 7), KSC_M, KSC_V),2);
        Sh =  1 + sum(repmat(rand(T,1) .* PS(:, 7), 1, 7 ) > PS, 2);
        
        % h
        dz = KSC_m(Sh);
        Omegaz_inv = spdiags(1./KSC_v2(Sh), 0, T, T);
        Sv_inv = spdiags([1/Vh, repmat(1/sigma2h, 1, T-1)]', 0, T, T);
        Kh = Omegaz_inv + H'*Sv_inv*H;
        hhat = Kh\(Omegaz_inv * (ytilde - dz) + [mh/Vh; zeros(T-1, 1)]);
        P = chol(Kh);
        h = hhat + P\randn(T, 1);
        
        % ystar
        Systar_inv = spdiags([exp(-g(1))/Vystar; exp(-g(2:T))], 0, T, T);
        Omegah_inv = spdiags(exp(-h), 0, T, T);
        Kystar = H' * Systar_inv * H + Omegah_inv + b(2)^2 / sigma2x * IT;
        ystarhat = Kystar\ ([mystar * exp(-g(1)) / Vystar; zeros(T-1, 1)] + Omegah_inv * y + b(2) / sigma2x * (x - b(1)));
        P = chol(Kystar);
        ystar = ystarhat + P\randn(T, 1);
    
        %sigma2h, sigma2g, sigma2z, sigma2x
        shhat = sh + sum(diff(h).^2); % h equation (y log volatility): error variance
        vhhat = vh + T - 1; % h equation (y log volatility): error variance
        sghat = sg + sum(diff(g).^2);  % g equation (ystar log volatility): error variance
        vghat = vg + T - 1;  % g equation (ystar log volatility): error variance
        szhat = sz + sum((logz-a(1)-a(2)*g).^2); % z equation (RV measurement): error variance
        vzhat = vz + T; % z equation (RV measurement): error variance
        sxhat = sx + sum((x-b(1)-b(2)*ystar).^2); % x equation (ystar measurement): error variance
        vxhat = vx + T; % x equation (ystar measurement): error variance
    
        sigma2h = 1/gamrnd(vhhat/2, 2/shhat);
        sigma2g = 1/gamrnd(vghat/2, 2/sghat);
        sigma2z = 1/gamrnd(vzhat/2, 2/szhat);
        sigma2x = 1/gamrnd(vxhat/2, 2/sxhat);
    
        % a: z equation (RV measurement) coefficients
        X(:,2) = g;
        Kahat = Ka + X' * X / sigma2z;
        mahat = Kahat\(Ka * ma + X'*logz /sigma2z);
        a = mahat + chol(Kahat)\randn(2,1);
    
        % b; x equation (ystar measurement) coefficients
        X(:,2) = ystar;
        Kbhat = Kb + X' * X / sigma2x;
        mbhat = Kbhat\(Kb * mb + X'*x /sigma2x);
        b = mbhat + chol(Kbhat)\randn(2,1);

        % save posterior
        if iter > Burnin
            ii = iter - Burnin;
            Post.sigma2h(ii) = sigma2h; %h
            Post.sigma2g(ii) = sigma2g; %g
            Post.sigma2z(ii) = sigma2z; %x
            Post.sigma2x(ii) = sigma2x; %x
            Post.a(ii,:) = a';
            Post.b(ii,:) = b';
            Post.g1(ii) = g(1);
            Post.h1(ii) = h(1);
            Post.ystar(:, ii) = ystar;
            Post.expg(:, ii) = exp(g/2); %std

            Post.ystar_pm= Post.ystar_pm + ystar;
            Post.expg_pm= Post.expg_pm + exp(g/2); %std
            Post.g_pm= Post.g_pm + g;
            Post.exph_pm= Post.exph_pm + exp(h/2); %std
            Post.h_pm= Post.h_pm + h;
        end
    end
    Post.ystar_pm = Post.ystar_pm / Nuse;
    Post.expg_pm = Post.expg_pm / Nuse;
    Post.g_pm = Post.g_pm / Nuse;
    Post.exph_pm = Post.exph_pm / Nuse;
    Post.h_pm = Post.h_pm / Nuse;

    % collect the last sample
    Post.last.sigma2h = sigma2h; % h equation (y log volatility): error variance
    Post.last.sigma2g = sigma2g; % g equation (ystar log volatility): error variance
    Post.last.sigma2z = sigma2z; % z equation (RV measurement): error variance
    Post.last.sigma2x = sigma2x; % x equation (ystar measurement): error variance
    Post.last.ystar = ystar; % ystar values
    Post.last.g = g; % h equation (y log volatility)
    Post.last.h = h; % g equation (ystar log volatility)
    Post.last.a = a; % z equation (RV measurement): coefficients
    Post.last.b = b; % x equation (ystar measurement): coefficients
end