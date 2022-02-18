function [V,D] = cgal_eig(X)
% Eig in Lanczos based LMO solver sometimes fall in to numerical issues. 
% This function replaces eig with a SVD based solver, in case eig does not
% converge. 
try
    [V,D]       = eig(X);
catch 
	warning('eig did not work, using svd based replacement');
    [V,D,W]     = svd(X);
    d           = diag(D).' .* sign(real(dot(V,W,1)));
    [d,ind]     = sort(d);
    D           = diag(d);
    V           = V(:,ind);
end