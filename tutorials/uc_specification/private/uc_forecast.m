function f = uc_forecast(m, out)
% uc_forecast - the predictive distributions of the models of uc_model other than
% 'ARbound' at each posterior draw of uc_sampler, for y_{T+1} and for the average of
% y_{T+1}, ..., y_{T+4}. Each draw simulates the log variances four quarters ahead from
% its parameters and its h_T and g_T; given those, both targets are normal, with the trend
% and the gap integrated out:
%   y_{T+1}:   mean tau_T + psi*u_T,    variance G_1 + H_1,
%   average:   mean tau_T + psi*u_T/4,  variance (16G_1 + 9G_2 + 4G_3 + G_4
%              + (1+psi)^2 (H_1 + H_2 + H_3) + H_4)/16,
% with G_j = exp(g_{T+j}) and H_j = exp(h_{T+j}).
%   f.m1, f.v1, f.m4, f.v4 : nsim x 1 means and variances of the two targets
%   Random numbers: randn(nsim,1) per quarter, the gap first, for each component
%   whose log variance is not constant.
n = size(out.phi, 1);
last = out.last;
[xg, j] = path(m.gap, out.phi, 0, last(:,2), n);
xt = path(m.trend, out.phi, j, last(:,3), n);
H = exp(xg); G = exp(xt);
if m.ma, psi = tanh(out.phi(:,end)); else, psi = zeros(n,1); end
tauT = last(:,1); uT = last(:,4);
f.m1 = tauT + psi.*uT;
f.v1 = G(:,1) + H(:,1);
f.m4 = tauT + psi.*uT/4;
f.v4 = (16*G(:,1) + 9*G(:,2) + 4*G(:,3) + G(:,4) ...
    + (1 + psi).^2.*sum(H(:,1:3), 2) + H(:,4))/16;
end

function [x, j] = path(k, phi, j, xT, n)
% the log variance in quarters T+1, ..., T+4 from the columns of phi after j
x = zeros(n, 4);
switch k.type
    case 'const'
        x = repmat(phi(:,j+1), 1, 4); j = j + 1;
    case 'rw'
        s = sqrt(exp(phi(:,j+2))); j = j + 2; prev = xT;
        for q = 1:4, prev = prev + s.*randn(n,1); x(:,q) = prev; end
    case 'ar1'
        mu = phi(:,j+1); rho = tanh(phi(:,j+2)); s = sqrt(exp(phi(:,j+3))); j = j + 3;
        prev = xT;
        for q = 1:4, prev = mu + rho.*(prev - mu) + s.*randn(n,1); x(:,q) = prev; end
end
end
