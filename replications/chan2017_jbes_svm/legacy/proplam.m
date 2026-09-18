% This function obtains a proposal density for sampling (mu, phi) using a 
% MH step
% Details: use the Newton-Raphason method (with BHHH) to maximize the 
% conditional distribution of (mu, phi) parameterized as 
% delta = (mu, tanh^(-1)(phi))
% outputs: a draw for (mu, phi) and the proposal density function

function [lam g] = proplam(h,sig2)
    
maxcount = 100;
T = size(h,1);
s = zeros(T,2);
mut = mean(h);
phit = (h(1:end-1)-mut)'*(h(2:end)-mut)/sum((h(1:end-1)-mut).^2);
delt = [mut atanh(phit)]';
count = 0;
flag = 0;
while flag == 0
    s(1,1) = sech(delt(2))^2/sig2*(h(1)-delt(1));
    s(2:T,1) = (1-tanh(delt(2)))/sig2 * ...
        (h(2:end)-tanh(delt(2))*h(1:end-1) - delt(1)*(1-tanh(delt(2))));    
    s(1,2) = sech(delt(2))^2*tanh(delt(2))/sig2 * (h(1)-delt(1));    
    s(2:T,2) = sech(delt(2))^2/sig2 * (h(1:end-1)-delt(1)) .* ...
        (h(2:T)-delt(1)-tanh(delt(2))*(h(1:end-1)-delt(1)));     
    S = sum(s)';
    B = s'*s;
    delt = delt + B\S;
    count = count + 1;
    if sum(abs(B\S)) > 10^(-4) && count < maxcount % stopping criteria
        flag = 1;
    end    
end
[C,p] = chol(B,'lower');
if count == maxcount || p~=0
    delt = [mut atanh(phit)]';
    s(1,1) = sech(delt(2))^2/sig2*(h(1)-delt(1));
    s(2:T,1) = (1-tanh(delt(2)))/sig2 * ...
        (h(2:end)-tanh(delt(2))*h(1:end-1) - delt(1)*(1-tanh(delt(2))));    
    s(1,2) = sech(delt(2))^2*tanh(delt(2))/sig2 * (h(1)-delt(1));    
    s(2:T,2) = sech(delt(2))^2/sig2 * (h(1:end-1)-delt(1)) .* ...
        (h(2:T)-delt(1)-tanh(delt(2))*(h(1:end-1)-delt(1)));    
    B = s'*s;
    C = chol(B,'lower');
end
% t proposal with df nu = 5
nu = 5;
del = delt + C'\randn(2,1)/sqrt(gamrnd(nu/2,2/nu));
lam = [del(1); tanh(del(2))]; 
c = gammaln((nu+2)/2) - gammaln(nu/2) - log(nu) - log(pi) + .5*log(det(B)); 
g = @(x) c + 2*log(cosh(atanh(x(2)))) ... % Jacobian of transformation 
    - (nu+2)/2 * log( 1 + 1/nu * ([x(1) atanh(x(2))]-delt')*B*([x(1); atanh(x(2))]-delt));
end