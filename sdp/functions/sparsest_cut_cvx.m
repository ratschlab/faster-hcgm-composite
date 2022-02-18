%code adapted from https://github.com/hwagyesa/scpsdp_alm.git
function [result, X_opt] = sparsest_cut_cvx(LAPL, show_plot, read_from_file, filename)
    EIG_TOL = 1e-4; % magnitude below this -> eigenvalue treated as zero
    results_folder = 'data/cvx_solutions/';
    path_to_file = strcat(results_folder, filename);
    if not(read_from_file)
        tstart = tic;
        n = size(LAPL, 1);
        cvx_begin sdp
%         cvx_solver mosek
        %cvx_precision high
    %     cvx_quiet true
            variable X(n,n) semidefinite
            minimize trace(LAPL * X)
            subject to
            n*trace(X) - trace(ones(n,n)*X) == 0.5 * n^2
%              (1/n) * trace(X) - (1/n^2) * trace(ones(n,n)*X) == 1
            % Triangle inequality constraints
            for i = 1:n
              for j = i+1:n
                for k = j+1:n
                  % These are for the standard problem
                  X(i,j) + X(j,k) - X(i,k) - X(j,j) <= 0
                  X(i,k) + X(j,k) - X(i,j) - X(k,k) <= 0
                  X(i,k) + X(i,j) - X(j,k) - X(i,i) <= 0
                end
              end
            end

            %Trace constraints
             trace(X) <= n - 1e-5;
        cvx_end
        
        [V, E] = eig(X);
        eigens = diag(E);
        eigens(abs(eigens) <= 1e-5) = 0;
        metric_mx = V(:, eigens ~= 0) * diag(sqrt(eigens(eigens ~= 0)));
        distance_mx = pdistmx(metric_mx').^2; %\ell_2^2 distance matrix
        fprintf('OPT_VAL: %f.\n', trace(LAPL*X)/n^2);
        % can also just take 2*ones(n,n) - 2*X

        figure('Name', 'Heatmap' , 'Color', 'white');
        subplot(2, 2, 1);
        heatmap(X);
        title('Result = X')


        subplot(2, 2, 2);
        heatmap(distance_mx);
        title('Result = distance matrix')


        subplot(2, 2, [3, 4]);
        heatmap(LAPL);
        title('Laplacian')
        
        fprintf('SDP OPT bound: %f.\n', trace(LAPL * X));
        result = trace(LAPL * X);
        X_opt = X;
        save (path_to_file, 'result', 'X');
        
    else
        solutions = load(path_to_file);
        result = solutions.result;
        X_opt = solutions.X;
    end
    
end