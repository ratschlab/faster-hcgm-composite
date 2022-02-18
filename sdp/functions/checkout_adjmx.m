%code adapted from https://github.com/hwagyesa/scpsdp_alm.git
function [graph_exists, A, D] = checkout_adjmx(graph_name, data_dir)
adj_mx_dir = '/adjacency_matrices';
thepath = strcat(data_dir, adj_mx_dir, '/', graph_name, '.mat');

graph_exists = exist(thepath, 'file');

if graph_exists
  % Load it
  data = load(thepath);
  A = data.A;
  D = data.D;
else
  A = 0;
  D = 0;
end