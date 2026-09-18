% ssm.diffmat - the sparse first-order difference matrix of a state equation,
% H = I_T - a*L, where L is the first subdiagonal (the lag operator on a stacked
% path).
%
%   H = ssm.diffmat(T)        % a = 1: the random-walk difference matrix
%   H = ssm.diffmat(T, a)
%
%   T : path length, a positive integer
%   a : scalar autoregressive coefficient (default 1)
%   H : T x T sparse lower bidiagonal matrix
%
% For x_t = a*x_{t-1} + u_t with u ~ N(0, S), H*x = u, so the path has precision
% H'*inv(S)*H, which is banded. Sign conventions, both of which appear in the
% papers:
%   AR(1) / random-walk state:  H_rho  = I - rho*L   ->  diffmat(T, rho)
%   MA(1) error transform:      H_psi  = I + psi*L   ->  diffmat(T, -psi)
%
% Code-identical to bvar.util.diffmat in bvar-toolkit apart from the error
% identifiers (ssm:diffmat:*), and the two must stay so (tests/unit/test_twins.m).

function H = diffmat(T, a)
if nargin < 2
    a = 1;
end
if ~isscalar(T) || T < 1 || T ~= round(T)
    error('ssm:diffmat:badT', 'T must be a positive integer');
end
if ~isscalar(a)
    error('ssm:diffmat:badA', 'a must be a scalar');
end
if T == 1
    H = speye(1);
    return
end
H = speye(T) - a*sparse(2:T, 1:(T-1), ones(1, T-1), T, T);
end
