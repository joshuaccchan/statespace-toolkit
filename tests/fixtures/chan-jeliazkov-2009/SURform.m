function Z = SURform(X)
% This funciton constructs sparse SUR/stacked design matrix
%
% Input:
% X: T-by-k design matrix, where row t is x_t'
%
% Output:
% Z: T-by-(T*k) sparse matrix diag(x_1', ..., x_T')

if ~isnumeric(X) || ~ismatrix(X)
    error('SURform:InputError', 'X must be a numeric 2-D matrix.');
end

[T, k] = size(X);

% row indices: each t repeated k times
row_idx = kron((1:T)', ones(k,1));

% column indices: 1,2,...,T*k (block for each t is consecutive k columns)
col_idx = (1:T*k)';

% values: stack rows of X as (x_1', x_2', ..., x_T')'
vals = reshape(X', T*k, 1);

Z = sparse(row_idx, col_idx, vals, T, T*k);
end