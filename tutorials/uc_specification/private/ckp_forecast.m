function f = ckp_forecast(r, yT)
% ckp_forecast - the predictive distributions of the bounded trend model of Chan,
% Koop and Potter (2013) at each posterior draw of ckp_run, for y_{T+1} and for the
% average of y_{T+1}, ..., y_{T+4}. Each draw simulates tau, rho and h four quarters
% ahead: tau and rho as random walks truncated to (0, 5) and (0, 1), h as a random
% walk. Given those paths the gap c_t = y_t - tau_t is a Gaussian AR(1) from
% c_T = yT - tau_T, so both targets are normal:
%   y_{T+1}:  mean tau_{T+1} + rho_{T+1}*c_T, variance exp(h_{T+1});
%   average:  mean of tau_{T+j} + P_j*c_T over j, with P_j = rho_{T+1}...rho_{T+j},
%             variance sum_k w_k^2 exp(h_{T+k}),
%             w_k = (1/4) sum_{j>=k} rho_{T+k+1}...rho_{T+j}.
%   f.m1, f.v1, f.m4, f.v4 : n x 1 means and variances of the two targets
n = size(r.sig, 1);
tau = zeros(n, 4); rho = zeros(n, 4); h = zeros(n, 4);
pt = r.last(:,1); pr = r.last(:,2); ph = r.last(:,3);
for q = 1:4
    pt = ssm.tnormrnd(pt, r.sig(:,1), 0, 5);
    pr = ssm.tnormrnd(pr, r.sig(:,2), 0, 1);
    ph = ph + sqrt(r.sig(:,3)).*randn(n,1);
    tau(:,q) = pt; rho(:,q) = pr; h(:,q) = ph;
end
cT = yT - r.last(:,1);
f.m1 = tau(:,1) + rho(:,1).*cT;
f.v1 = exp(h(:,1));
P = cumprod(rho, 2);
f.m4 = mean(tau, 2) + mean(P, 2).*cT;
f.v4 = zeros(n, 1);
for k = 1:4
    w = zeros(n, 1); prodk = ones(n, 1);
    for j = k:4
        if j > k, prodk = prodk.*rho(:,j); end
        w = w + prodk;
    end
    f.v4 = f.v4 + (w/4).^2.*exp(h(:,k));
end
end
