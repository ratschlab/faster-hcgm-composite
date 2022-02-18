function result = lmo_psd_weight(X, kappa)
    % this is supposed to receive a linearized nxn symmetric matrix
    % and return the lmo of it over the PSD cone intersected with 
    % ball of matrices with trace norm <= kappa
    
    square_size = floor(sqrt(length(X)));
    assert(square_size^2 == length(X));
    X = reshape(X, [square_size, square_size]);
    X = (X + X')/2;
    
    c1 = 2;
    c2 = 1;
    t = 10;
    
    [V, d, ~] = ApproxMinEvecLanczos(X, square_size, ceil(c1 * (t^0.5) * log(c2 * square_size)));
    V = V / norm(V, 2);
    
%     [V, d] = eigs(X, 1, 'smallestreal', 'Tolerance', 1e-9);
%     %V = V / norm(V, 2);
    
    if d < 0
        result = kappa * (V * V'); % do I need this normalization?
    else
        result = zeros(square_size, square_size);
    end

    result = result(:);
end