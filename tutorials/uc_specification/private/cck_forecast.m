function f = cck_forecast(out, piT)
% cck_forecast - the predictive distributions of inflation under M1 of Chan, Clark and
% Koop (2018) at each posterior draw of cck_sampler, for pi_{T+1} and for the average
% of pi_{T+1}, ..., pi_{T+4}. Each draw simulates b (truncated to (0,1)) and the two
% log-volatilities four quarters ahead; given those, both targets are normal, with
% the trend innovations and the gap integrated out. With c_T = piT - pistar_T,
% N_j = exp(ln_{T+j}), V_j = exp(lv_{T+j}) and P_j = b_{T+1}...b_{T+j}:
%   pi_{T+1}:  mean pistar_T + b_{T+1} c_T,  variance N_1 + V_1;
%   average:   mean pistar_T + c_T mean_j P_j,  variance (16N_1 + 9N_2 + 4N_3 + N_4)/16
%              + sum_k (w_k/4)^2 V_k,  w_k = sum_{j>=k} b_{T+k+1}...b_{T+j}.
% forecast_M1.m of the paper draws the trend path as well and uses the same formulas
% given it.
%   f.m1, f.v1, f.m4, f.v4 : nsim x 1 means and variances of the two targets
n = size(out.last, 1);
sigb2 = out.theta(:,8); phiv = out.theta(:,10); phin = out.theta(:,11);
b = zeros(n, 4); lv = zeros(n, 4); ln = zeros(n, 4);
pb = out.last(:,2); pv = out.last(:,3); pn = out.last(:,4);
for q = 1:4
    pv = pv + sqrt(phiv).*randn(n,1);
    pn = pn + sqrt(phin).*randn(n,1);
    pb = ssm.tnormrnd(pb, sigb2, 0, 1);
    lv(:,q) = pv; ln(:,q) = pn; b(:,q) = pb;
end
N = exp(ln); V = exp(lv);
cT = piT - out.last(:,1);
f.m1 = out.last(:,1) + b(:,1).*cT;
f.v1 = N(:,1) + V(:,1);
f.m4 = out.last(:,1) + mean(cumprod(b, 2), 2).*cT;
f.v4 = (16*N(:,1) + 9*N(:,2) + 4*N(:,3) + N(:,4))/16;
for k = 1:4
    w = zeros(n, 1); pk = ones(n, 1);
    for j = k:4
        if j > k, pk = pk.*b(:,j); end
        w = w + pk;
    end
    f.v4 = f.v4 + (w/4).^2.*V(:,k);
end
end
