% DFM.m
% Dynamic factor model with AR(1) factor dynamics, applied to FRED-MD
% macroeconomic series:
%
%     y_t = A * f_t + eps_t,        eps_t ~ N(0, Sigma),  Sigma = diag(sig2_1, ..., sig2_n)
%     f_t = Phi * f_{t-1} + u_t,    u_t   ~ N(0, Omega),  Omega = diag(omega2_1, ..., omega2_r)
%
% where A is the n-by-r factor loading matrix (lower-triangular
% normalization with unit diagonal in the top r-by-r block) and Phi is
% diagonal so that each factor follows an independent AR(1). Priors:
%
%     a_ij      ~ N(a0, Va)        (free elements of A)
%     phi_j     ~ N(phi0_j, Vphi_j) truncated to (-1, 1)
%     sig2_i    ~ IG(nusig2, Ssig2_i)
%     omega2_j  ~ IG(nuomega2, Somega2_j)
%
% The latent factor path (f_1, ..., f_T) is sampled in one block using
% the precision sampler of Chan and Jeliazkov (2009).

clear; clc; rng(42);

nsim = 20000;
burnin = 1000;
r = 1; % number of factors

% load data
raw = readtable('FRED-MD.csv','VariableNamingRule','preserve');
dates_num = raw{:,1}; % first column contains dates as fractional years
year_part = floor(dates_num);
month_part = 1 + round(12*(dates_num - year_part));
dates = datetime(year_part, month_part, 1);
data = raw{:,2:end}; % remaining columns are data
varnames = raw.Properties.VariableNames(2:end);

% move the 6th column (INDPRO) to the first column
perm = [6, 1:5, 7:size(data,2)];
data = data(:,perm);
varnames = varnames(perm);

% remove columns with missing values
idx = all(~isnan(data),1);
data = data(:,idx);
varnames = varnames(idx);

% standardize the data
data_mean = mean(data,1)';
data_std = std(data,0,1)';
Y = (data - data_mean') ./ data_std';

[T,n] = size(Y);
y = reshape(Y',T*n,1);

% storage
store_F = zeros(nsim,T,r);
store_A = zeros(nsim,n,r);
store_sig2 = zeros(nsim,n);
store_omega2 = zeros(nsim,r);
store_phi = zeros(nsim,r);

% prior hyperparameters
a0 = 0; Va = 1; % a_ij iid N(a0,Va)
phi0 = zeros(r,1); Vphi = ones(r,1);
nusig2 = 3; Ssig2 = (nusig2-1)*ones(n,1);
nuomega2 = 3; Somega2 = (nuomega2-1)*ones(r,1);

% initialize the Markov chain
sig2 = var(Y)';
omega2 = ones(r,1);
phi = 0.5*ones(r,1);
Phi = spdiags(phi,0,r,r);
A = [eye(r); zeros(n-r,r)];   % lower-triangular normalization

% matrices used to build H_phi
hzeros = sparse(r,(T-1)*r);
vzeros = sparse(T*r,r);
HPhi = speye(T*r) - cat(2,cat(1,hzeros,kron(speye(T-1),Phi)),vzeros);

for isim = 1:(nsim + burnin)
    % sample f
    iSig = spdiags(1./sig2,0,n,n);
    iOmega = spdiags(repmat(1./omega2,T,1),0,T*r,T*r);
    Kf = HPhi' * iOmega * HPhi + kron(speye(T), A' * iSig * A);
    f_hat = Kf \ (kron(speye(T), A' * iSig) * y);
    f = f_hat + chol(Kf,'lower')' \ randn(T*r,1);
    F = reshape(f,r,T)';

    % sample A equation by equation
    for ieq = 2:n
        nai = min(ieq-1,r);
        Xf = F(:,1:nai);
        K_ai = spdiags((1/Va)*ones(nai,1),0,nai,nai) + (Xf' * Xf) / sig2(ieq);
        if ieq <= r
            ai_hat = K_ai \ ((a0/Va)*ones(nai,1) + Xf' * (Y(:,ieq) - F(:,ieq)) / sig2(ieq));
        else
            ai_hat = K_ai \ ((a0/Va)*ones(nai,1) + Xf' * Y(:,ieq) / sig2(ieq));
        end
        A(ieq,1:nai) = ai_hat + chol(K_ai,'lower')' \ randn(nai,1);
    end

    % sample sig2
    E_y = Y - F * A';
    sig2 = 1 ./ gamrnd(nusig2 + T/2, 1 ./ (Ssig2 + sum(E_y.^2)'/2));

    % sample omega2
    E_f = [F(1,:); F(2:end,:) - F(1:end-1,:) * Phi];
    omega2 = 1 ./ gamrnd(nuomega2 + T/2, 1 ./ (Somega2 + sum(E_f.^2)'/2));

    % sample phi equation by equation
    Zf = [zeros(1,r); F(1:end-1,:)];
    for jj = 1:r
        Kphi_j = 1/Vphi(jj) + sum(Zf(:,jj).^2) / omega2(jj);
        phi_hat_j = (phi0(jj)/Vphi(jj) + sum(Zf(:,jj).*F(:,jj)) / omega2(jj)) / Kphi_j;
        phi_sd_j = sqrt(1/Kphi_j);
        accepted = false;
        while ~accepted
            phi_prop = phi_hat_j + phi_sd_j * randn;
            if abs(phi_prop) < 1
                phi(jj) = phi_prop;
                accepted = true;
            end
        end
    end
    Phi = spdiags(phi,0,r,r);
    HPhi = speye(T*r) - cat(2,cat(1,hzeros,kron(speye(T-1),Phi)),vzeros);

    if isim > burnin
        isave = isim - burnin;
        store_F(isave,:,:) = F;
        store_A(isave,:,:) = A;
        store_sig2(isave,:) = sig2';
        store_omega2(isave,:) = omega2';
        store_phi(isave,:) = phi';
    end

    if mod(isim,5000) == 0
        fprintf('Iteration %d of %d (%.1f%%)\n', ...
            isim, nsim + burnin, 100*isim/(nsim + burnin));
    end
end
F_mean = reshape(mean(store_F,1),T,r);

% NBER recession dates
rec_starts = [ ...
    datetime(1960,4,1)
    datetime(1969,12,1)
    datetime(1973,11,1)
    datetime(1980,1,1)
    datetime(1981,7,1)
    datetime(1990,7,1)
    datetime(2001,3,1)
    datetime(2007,12,1)
    datetime(2020,2,1)];

rec_ends = [ ...
    datetime(1961,2,1)
    datetime(1970,11,1)
    datetime(1975,3,1)
    datetime(1980,7,1)
    datetime(1982,11,1)
    datetime(1991,3,1)
    datetime(2001,11,1)
    datetime(2009,6,1)
    datetime(2020,4,1)];

% full-sample and pre-COVID plots of the posterior mean factor
idx_pre = dates <= datetime(2019,12,1);
date_pre_end = dates(find(idx_pre,1,'last'));

figure;
if r == 1
    % single-factor layout: full sample (top), pre-COVID (bottom)
    subplot(2,1,1);
    draw_factor_panel(dates, F_mean(:,1), rec_starts, rec_ends, ...
        'Full sample', 'Factor', '', true, 14);

    subplot(2,1,2);
    draw_factor_panel(dates(idx_pre), F_mean(idx_pre,1), rec_starts, rec_ends, ...
        'Pre-COVID sample', 'Factor', 'Date', false, 14);

    set(gcf, 'Position', [100 100 900 600]);
else
    % multi-factor layout: r rows x 2 cols, each row is one factor
    for j = 1:r
        xlab_j = ''; if j == r; xlab_j = 'Date'; end

        subplot(r,2,2*(j-1)+1);
        draw_factor_panel(dates, F_mean(:,j), rec_starts, rec_ends, ...
            sprintf('Factor %d: full sample', j), sprintf('Factor %d', j), xlab_j, j == 1, 12);

        subplot(r,2,2*(j-1)+2);
        draw_factor_panel(dates(idx_pre), F_mean(idx_pre,j), rec_starts, rec_ends, ...
            sprintf('Factor %d: pre-COVID sample', j), '', xlab_j, false, 12);
    end
    set(gcf, 'Position', [100 100 1200 max(300, 250*r)]);
end


% --- local function ---
function draw_factor_panel(dates_x, F_x, rec_starts, rec_ends, ttl, ylab, xlab, show_legend, fontsize)
    yl = [min(F_x) - 0.2, max(F_x) + 0.2];
    date_end = dates_x(end);
    hold on;
    h_first_rec = [];
    for i = 1:length(rec_starts)
        if rec_starts(i) <= date_end
            rec_end_clipped = min(rec_ends(i), date_end);
            h_rec = patch([rec_starts(i) rec_end_clipped rec_end_clipped rec_starts(i)], ...
                          [yl(1) yl(1) yl(2) yl(2)], ...
                          [0.92 0.92 0.92], 'EdgeColor','none');
            uistack(h_rec,'bottom');
            if isempty(h_first_rec)
                h_first_rec = h_rec;
            end
        end
    end
    h_factor = plot(dates_x, F_x, 'k', 'LineWidth', 1.5);
    yline(0, 'k');
    hold off;
    box off;
    ax = gca;
    ax.Layer = 'top';
    ax.FontSize = fontsize;
    xlim([dates_x(1) date_end]);
    ylim(yl);
    title(ttl);
    if ~isempty(ylab); ylabel(ylab); end
    if ~isempty(xlab); xlabel(xlab); end
    if show_legend && ~isempty(h_first_rec)
        legend([h_factor, h_first_rec], {'Posterior mean factor', 'NBER recessions'}, ...
               'Location', 'best', 'Box', 'off');
    end
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
end
