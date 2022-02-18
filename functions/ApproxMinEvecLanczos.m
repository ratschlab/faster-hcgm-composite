function [v, xi, i] = ApproxMinEvecLanczos(M, n, q)
% Approximate minimum eigenvector
% Vanilla Lanczos method

q = min(q, n-1);                    % Iterations < dimension!

if isnumeric(M)
    M = @(x) M*x;
end

Q = zeros(n, q+1);                  % Lanczos vectors

aleph = zeros(q,1);                 % Diagonal Lanczos coefs
beth = zeros(q,1);                  % Off-diagonal Lanczos coefs

Q(:,1) = randn(n, 1);               % First Lanczos vector is random
Q(:,1) = Q(:,1) / norm(Q(:,1));

for i = 1 : q
    Q(:, i+1) = M ( Q(:, i) );				% Apply M to previous Lanczos vector
    aleph(i) = real(Q(:, i)' * Q(:, i+1));		% Compute diagonal coefficients
    
    if (i == 1)                     % Lanczos iteration
        Q(:, i+1) = Q(:, i+1) - aleph(i) * Q(:, i);
    else
        Q(:, i+1) = Q(:, i+1) - aleph(i) * Q(:, i) - beth(i-1) * Q(:, i-1);
    end % if
    
    beth(i) = norm( Q(:, i+1) );            % Compute off-diagonal coefficients
    
    if ( abs(beth(i)) < sqrt(n)*eps ), break; end
    
    Q(:, i+1) = Q(:, i+1) / beth(i);        % Normalize
end % for i

% i contains number of completed iterations

B = diag(aleph(1:i), 0) + diag(beth(1:(i-1)), +1) + diag(beth(1:(i-1)), -1);

[U, D] = cgal_eig(B+B');
[xi, ind] = min(diag(D));
v = Q(:, 1:i) * U(:, ind);
end % function