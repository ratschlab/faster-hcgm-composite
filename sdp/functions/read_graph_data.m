%code adapted from https://github.com/hwagyesa/scpsdp_alm.git
function LAPL = read_graph_data(graph_name)
    root_dir = '';
    data_dir = strcat(root_dir, 'data');

    tstart = tic;

    [graph_exists, A, D] = checkout_adjmx(graph_name, data_dir);

    if graph_exists
      % Adjacency matrix exists, and we loaded it
      fprintf('(time %f) Adjacency matrix for graph %s loaded.\n', toc(tstart), graph_name);
    else
      % No adjacency matrix exists. Create it from the .graph file
      fprintf('(time %f) Adjacency matrix for graph %s doesn''t exist; creating it.\n',...
        toc(tstart), graph_name);
      [A, D] = graph_to_adjmx(graph_name, data_dir);
      checkin_adjmx(graph_name, data_dir, A, D);
    end
    
    % from SPARSEST CUT problem formuation in pg 388 of approximation algorithms
    % and this post: https://math.stackexchange.com/questions/2288457/sdp-relaxation-for-the-sparset-cut
    LAPL = full(D - A); 
end