%code adapted from https://github.com/hwagyesa/scpsdp_alm.git
function [D] = pdistmx(X)
n = size(X,2);
d = size(X,1);
for i = 1:n
  for j = 1:n
    D(i,j) = norm(X(:,i) - X(:,j), 2);%.^2 * 1/2;
  end
end

return