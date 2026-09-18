% % =======================================================================
% % support function: maximize the conditional distribution of the factor 
% % loadings a marginal of the factors f 
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

function [at ahat invDa] = maxA(fstar,Y, bigGbeta,invVa,invSig1,q,id)

n = size(invSig1,1);
T = length(fstar)/q;
k = size(invVa,1);
Tn = T*n;
shortfstar = reshape(fstar,q,T)';
bigF = zeros(Tn,k); 
countf = 1;
for i=2:n
    minq = min(i-1,q);
    bigF(i:n:end,countf:countf+minq-1) = shortfstar(:,1:minq);
    countf = countf+minq;
end
sparseF = sparse(bigF);
bigFinvSig1 = sparseF'*kron(speye(T,T), invSig1);
invDa = invVa + bigFinvSig1*sparseF;
da = bigFinvSig1*(Y-bigGbeta - reshape([shortfstar(:,1:q) zeros(T,n-q)]',Tn,1));
ahat = invDa\da;
    
%% maximization
at = ahat;
err=ones(q,1);    
At = [eye(q); zeros(n-q,q)]; 
countloop = 0;
while norm(err)>10^(-3) && countloop<100
    At(id)=at;
    grad = invDa*(ahat-at);
    invSig1At = invSig1*At;
    AtinvSig1 = At'*invSig1;
    AtinvSig1At = At'*invSig1At;
    count = 1;
    for i=2:n
        for j=1:min(q,i-1)
            grad(count) = grad(count) - T/2*trace(AtinvSig1At\(sparse(i,j,1,n,q)'*invSig1At + AtinvSig1*sparse(i,j,1,n,q)));
            count = count+1;
        end        
    end
    I = invDa;
    err = I\grad;
    at = at + err;
    countloop = countloop+1;
end
if countloop>=100
    at = ahat;
end
end
