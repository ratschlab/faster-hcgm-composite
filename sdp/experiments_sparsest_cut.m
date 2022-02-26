%% Description
% This script implements the sparseset cut experiment.
%
% Based on code provided in [1]
%
% [1] Vladarean, Maria-Luiza, Ahmet Alacaoglu, Ya-Ping Hsieh, and Volkan Cevher.
% "Conditional gradient methods for stochastically constrained convex
% minimization." In International Conference on Machine Learning, PMLR, 2020.
% https://proceedings.mlr.press/v119/vladarean20a.html
%
% contact: Gideon Dresdner - dgideon@inf.ethz.ch

clear variables; 
clear global;
clear functions;
close all;
rng(22);
addpath('sparsest_cut_cvx_solvers')
addpath('functions')
addpath('functions/sparsest_cut_functions')
addpath('sp_cut_results')


%% Problem Construction
graph_name = "55n-insecta-ant-colony1-day37";
graph_name = "102n-insecta-ant-colony4-day10";
graph_name = "25mammalia-primate-association-13";

graph_results_folder = strcat('sp_cut_results/', graph_name, '/'); 
if ~exist(graph_results_folder, 'dir')
       mkdir(graph_results_folder)
end

% feel like XIJ should also be positive in the constraints
global LAPL;
LAPL = read_graph_data(graph_name);
%LAPL = LAPL/ norm(LAPL, 'fro'); % normalization

global Laplacian_vec;
Laplacian_vec = LAPL(:);

global d; 
d = size(LAPL, 1);

global nchoosek_inds;
nchoosek_inds = nchoosek(1 : d, 3); % matrix with 3 columns, containing the indices
total_constraints = size(nchoosek_inds, 1);

global rand_gradf_indices;
rand_gradf_indices = nchoosek(1 : d, 2);
main_diag = ones(d, 2) .* [1 : d]';
rand_gradf_indices = [rand_gradf_indices; main_diag];

global total_stoc_obj_fc;
total_stoc_obj_fc = size(rand_gradf_indices, 1);

global test_err_data;
test_err_data.compute = false;

global opt_val;
global X_true;
read_from_file = true  % change back to false
show_plots = false;
[opt_val, X_true] = sparsest_cut_cvx(LAPL, show_plots, read_from_file, graph_name);
opt_val
pause(50)
  
% opt_val = Problem.opt_val
% % figure(1)
% % imagesc(Problem.X_opt)
% % pause(50)
% global num_constraints;
% num_constraints = ;
% clearvars Problem

global trace_bound;
trace_bound = d;

global constant_gradient_f;
constant_gradient_f = LAPL; 
constant_gradient_f = constant_gradient_f(:);

global total_stoc_fcs;
total_stoc_fcs = size(nchoosek_inds, 1);

global batch_factor_mokhtari;
batch_factor_mokhtari = 0.05;


%% HELPER VARS
method_names = {};
max_it_lmo = 2e6;
max_it_sfo = 2e6;
i = 1;
results.conv_obj_lmo = zeros(max_it_lmo, 3);
results.conv_obj_sfo = zeros(max_it_sfo, 3);
results.conv_constr_lmo = zeros(max_it_lmo, 3);
results.conv_constr_sfo = zeros(max_it_sfo, 3);
results.last_idx_minib_lmo = zeros(max_it_lmo, 3);
results.last_idx_minib_sfo = zeros(max_it_sfo, 3);
results.optimal_Xs = zeros(d^2, 3);
results.test_err = zeros(1, 3);

[V, ~] = eigs(LAPL, 1);
X0 = trace_bound * (V * V');
%trace(X0)
X0 = X0(:);

last_epoch_with_part_batch = -1;
global_epoch_count = 5;%1e4;%5e4;

%% SHCGM
start_exp = tic; 
batch_fc = floor(batch_factor_mokhtari * total_stoc_obj_fc);
T_eps = global_epoch_count;
betas = [1e2];% 1e2 deemed optimal
for j = 1 : length(betas)
    start_exp = tic; 
    beta0=betas(j);
    [conv_obj_lmo, conv_obj_sfo, conv_constr_lmo, conv_constr_sfo, X_opt, ~] = ...
                            shcgm(@f, total_stoc_obj_fc, batch_fc, @grad_f, @grad_stoc_fc_shcgm, @feasibility, @lmo_func, X0, beta0, T_eps, test_err_data);
    fprintf("Time taken = %f seconds \n\n\n", ceil(toc(start_exp))); 
    method_names{end+1} = sprintf('shcgm-beta0-%1.5e', beta0);
    results.conv_obj_lmo(1 : length(conv_obj_lmo), i) = conv_obj_lmo;
    results.conv_obj_sfo(1 : length(conv_obj_sfo), i) = conv_obj_sfo;
    results.conv_constr_lmo(1 : length(conv_constr_lmo), i) = conv_constr_lmo;
    results.conv_constr_sfo(1 : length(conv_constr_sfo), i) = conv_constr_sfo;
    results.last_idx_minib_lmo(:, i) = -1;
    results.last_idx_minib_sfo(:, i) = -1;
    results.optimal_Xs(:, i) = X_opt;
    i = i + 1;
    max_it_lmo = length(conv_obj_lmo);
    max_it_sfo = length(conv_obj_sfo);
    result_file = strcat(graph_results_folder, sprintf('shcgm_%s_beta0_%.0e_iter_%d_batchfactor_%.3f.mat', datestr(now,'ddmm_HHMMSS'), beta0, T_eps, batch_factor_mokhtari));
    save(result_file, 'conv_obj_lmo', 'conv_obj_sfo', 'conv_constr_lmo', 'conv_constr_sfo', 'X_opt');
end


%% H-1SFW
betas = [1e-2]%, 5e-3 seems to do the job
batch_fc = floor(total_stoc_obj_fc * batch_factor_mokhtari);
batch_triang = floor(total_constraints * batch_factor_mokhtari);
T_eps = (total_constraints / batch_triang) * global_epoch_count;
for j = 1 : length(betas)
    start_exp = tic; 
    beta0=betas(j);
    start_exp = tic; 
    beta0 = betas(j);
    %The total_stoc_fcs needs to be d^2 here
    [conv_obj_lmo, conv_obj_sfo, conv_constr_lmo, conv_constr_sfo, X_opt, ~] = ...
                            mokhtari_as_constr(@f, total_constraints, [batch_fc, batch_triang], @grad_f, @grad_stoc_full_mokhtari, @feasibility, @lmo_func, X0, beta0, T_eps, test_err_data);
    fprintf("Time taken = %f seconds \n\n\n", ceil(toc(start_exp))); 
    method_names{end+1} = sprintf('mokhtari-as-full-beta0-%5.2e', beta0);
    results.conv_obj_lmo(1 : length(conv_obj_lmo), i) = conv_obj_lmo;
    results.conv_obj_sfo(1 : length(conv_obj_sfo), i) = conv_obj_sfo;
    results.conv_constr_lmo(1 : length(conv_constr_lmo), i) = conv_constr_lmo;
    results.conv_constr_sfo(1 : length(conv_constr_sfo), i) = conv_constr_sfo;
    results.last_idx_minib_lmo(:, i) = -1;
    results.last_idx_minib_sfo(:, i) = -1;
    results.optimal_Xs(:, i) = X_opt;
    i = i + 1;
    if max_it_lmo < length(conv_obj_lmo)
        max_it_lmo = length(conv_obj_lmo);
    end
    if max_it_sfo < length(conv_obj_sfo)
        max_it_sfo = length(conv_obj_sfo);
    end
    result_file = strcat(graph_results_folder, sprintf('mokhtari_full_%s_beta0_%.0e_iter_%d_batchfactor_%.3f.mat', datestr(now,'ddmm_HHMMSS'), beta0, T_eps, batch_factor_mokhtari));
    save(result_file, 'conv_obj_lmo', 'conv_obj_sfo', 'conv_constr_lmo', 'conv_constr_sfo', 'X_opt');
end

%% H-SPIDER-FW 
betas = [1e1]; %5e0 is not bad either. these don't work: 1e-3, 1e-1, 1e0. 1e1 seems best
for j = 1 : length(betas)
    start_exp = tic; 
    beta0 = betas(j);
    T_eps = 10;
    while get_mem_for_sfw(T_eps, total_constraints, 1) < global_epoch_count
        T_eps = T_eps + 1;
    end
    %   
    % fprintf('--- Starting spiderfw with T_eps = %d, with total_epochs = %d \n', T_eps, get_mem_for_sfw(T_eps, total_stoc_fcs, 10010));
    %T_eps = 20;
    [conv_obj_lmo, conv_obj_sfo, conv_constr_lmo, conv_constr_sfo, X_opt, last_idx_minib_lmo, last_idx_minib_sfo, last_epoch_with_part_batch, ~]...
                    = spider_fw(@f, total_constraints, @grad_F, @var_red_grad_F, @lmo_func, @feasibility, X0, beta0, T_eps, global_epoch_count, test_err_data);

%                     = spider_fw(@f, total_constraints, @var_red_grad_F, @lmo_func, @feasibility, X0, beta0, T_eps, global_epoch_count, test_err_data);
%                       spider_fw(f, total_stoc_fcs, grad_F, var_red_grad, lmo, feasibility, X0, beta0, T_eps, max_epochs, test_err_data)

                
    fprintf("Time taken = %f seconds \n\n\n", ceil(toc(start_exp))); 
    method_names{end+1} = sprintf('spider-fw-homotopy-beta0-%1.3e', beta0);
    results.conv_obj_lmo(1 : length(conv_obj_lmo), i) = conv_obj_lmo;
    results.conv_obj_sfo(1 : length(conv_obj_sfo), i) = conv_obj_sfo;
    results.conv_constr_lmo(1 : length(conv_constr_lmo), i) = conv_constr_lmo;
    results.conv_constr_sfo(1 : length(conv_constr_sfo), i) = conv_constr_sfo;
    results.last_idx_minib_lmo(1 : length(last_idx_minib_lmo), i) = last_idx_minib_lmo;
    results.last_idx_minib_sfo(1 : length(last_idx_minib_sfo), i) = last_idx_minib_sfo; 
    results.optimal_Xs(:, i) = X_opt;
    i = i + 1;
    max_it_lmo = length(conv_obj_lmo);
    max_it_sfo = length(conv_obj_sfo);
    result_file = strcat(graph_results_folder, sprintf('spiderfw_%s_beta0_%.0e_iter_%d.mat', datestr(now,'ddmm_HHMMSS'), beta0, T_eps));
    save(result_file, 'conv_obj_lmo', 'conv_obj_sfo', 'conv_constr_lmo', 'conv_constr_sfo', ...
                                                'last_idx_minib_lmo', 'last_idx_minib_sfo', 'X_opt');
end


%% Ploting
num_methods = length(method_names);
plot_convergence(...
    method_names(1:num_methods), results.conv_obj_lmo(1: max_it_lmo, 1:num_methods), ...
    results.conv_obj_lmo(1: max_it_lmo, 1:num_methods), results.conv_obj_sfo(1 : max_it_sfo, 1:num_methods),...
    results.conv_constr_lmo(1: max_it_lmo, 1:num_methods), results.conv_constr_sfo(1 : max_it_sfo, 1:num_methods), ...
    results.optimal_Xs(:, 1:num_methods), X_true, results.last_idx_minib_lmo(:, :), results.last_idx_minib_sfo(:, :), results.test_err, ...
    last_epoch_with_part_batch, opt_val, total_constraints);




%% FUNCTIONS

function result = f(X_lin)
    global Laplacian_vec;
    result = dot(Laplacian_vec, X_lin);
end

function result = grad_f(X_lin)
    global constant_gradient_f;
    result = constant_gradient_f;
end

function result = grad_stoc_f(batch_size, X_lin)
    global rand_gradf_indices;
    global constant_gradient_f;
    global d;
    
    rand_ind = randperm(size(rand_gradf_indices, 1), batch_size);
    
    result = zeros(length(constant_gradient_f), 1);
    
    inds = sub2ind([d, d], rand_gradf_indices(rand_ind, 1), rand_gradf_indices(rand_ind, 2));
    result(inds) = constant_gradient_f(inds);
    
    inds_sym = sub2ind([d, d], rand_gradf_indices(rand_ind, 2), rand_gradf_indices(rand_ind, 1));
    result(inds_sym) = constant_gradient_f(inds_sym);
end

function [grad_val, feas_eq, feas_ineq, norm_vk_minus_c] = grad_F(X_lin, beta)
    global LAPL;
    global d;
    global nchoosek_inds;
    
    X = reshape(X_lin, [d,d]);
    
    % Add the equality constraint
    [grad_val, displacement] = get_equality_constr_grad(X, beta);
    feas_eq = abs(displacement);
   
    % Add the triangle constraints. This needs to be scaled by d to match
    % the order of the equality constr
    n = length(nchoosek_inds);
    feas_ineq = 0;
    %mock_grad = zeros(size(grad_val));
    for l = 1 : n
        index_triplet = nchoosek_inds(l, :);
        [update_vector, row_idxs, col_idxs, part_feas_normsq] = get_triangle_constr_grad(X, beta, index_triplet);
        
        inds_to_update = sub2ind([d, d], row_idxs, col_idxs);
        grad_val(inds_to_update) = grad_val(inds_to_update) + update_vector;
        %mock_grad(inds_to_update) = mock_grad(inds_to_update) + update_vector;
        
        % symmetize update
        inds_to_update = sub2ind([d, d], col_idxs, row_idxs);
        grad_val(inds_to_update) = grad_val(inds_to_update) + update_vector;
        %mock_grad(inds_to_update) = mock_grad(inds_to_update) + update_vector;
        
        feas_ineq = feas_ineq + part_feas_normsq;
    end
    
    % FC_GRAD
    grad_val = grad_val + LAPL; % this is grad_f, supposed to be symm
%     fprintf('\n \n -------------------------');
%     fprintf('\n--norm_grad_f = %f, norm_eq_upd = %f, norm_ineq_upd = %f \n\n', ...
%         norm(LAPL, 'fro'), norm((d * eye(d, d) - ones(d, d)) * displacement, 'fro'), norm(mock_grad, 'fro'));
     
    feas_ineq = sqrt(feas_ineq);
    grad_val = grad_val(:);
    norm_vk_minus_c = -1;
end

function [d_k, full_grad_constr, useless2, useless3] = grad_stoc_fc_shcgm(batch_fc, X_lin, beta, rho_k, d_k_prev)
    global d;
    global nchoosek_inds;
    global total_stoc_obj_fc;

    % EQUALITY_CONSTR
    X = reshape(X_lin, [d, d]);
    [full_grad_constr, ~] = get_equality_constr_grad(X, beta);
    
    % TRIANGLE_CONSTR. This needs to be scaled by d to match
    % the order of the equality constr
    n = length(nchoosek_inds);
    %mock_grad = zeros(size(grad_val));
    for l = 1 : n
        index_triplet = nchoosek_inds(l, :);
        [update_vector, row_idxs, col_idxs, ~] = get_triangle_constr_grad(X, beta, index_triplet);
        
        inds_to_update = sub2ind([d, d], row_idxs, col_idxs);
        full_grad_constr(inds_to_update) = full_grad_constr(inds_to_update) +  update_vector;
        %mock_grad(inds_to_update) = mock_grad(inds_to_update) + update_vector;
        
        % symmetize update
        inds_to_update = sub2ind([d, d], col_idxs, row_idxs);
        full_grad_constr(inds_to_update) = full_grad_constr(inds_to_update) + update_vector;
        %mock_grad(inds_to_update) = mock_grad(inds_to_update) + update_vector;
    end

    % STOC_FC
    
    full_grad_constr = full_grad_constr(:);  
    
    stoc_gr_f = ( total_stoc_obj_fc / batch_fc) * grad_stoc_f(batch_fc, X_lin);
    d_k = (1 - rho_k ) * d_k_prev + rho_k * stoc_gr_f; % add stochastic fc
    
    useless2 = 0;
    useless3 = false;
end

function [grad_val, normX, norm_A, is_full] = grad_stoc_full_mokhtari(batches, X_lin, beta, rho_k, d_k_prev)
    global d;
    global nchoosek_inds;
    global total_stoc_obj_fc;
    
    batch_fc = batches(1);
    batch_triang = batches(2);

    % EQUALITY_CONSTR - grad = rhoK * constr1
    X = reshape(X_lin, [d, d]);
    [grad_val, ~] = get_equality_constr_grad(X, beta);
    grad_val = rho_k * grad_val;
    
    % TRIANGLE_CONSTR. - grad += rhok * constr2
    n = length(nchoosek_inds);
    rand_triang = randperm(n, batch_triang);
    for l = 1 : batch_triang
        index_triplet = nchoosek_inds(rand_triang(l), :);
        [update_vector, row_idxs, col_idxs, ~] = get_triangle_constr_grad(X, beta, index_triplet);
        
        inds_to_update = sub2ind([d, d], row_idxs, col_idxs);
        grad_val(inds_to_update) = grad_val(inds_to_update) + rho_k * (n / batch_triang) * update_vector;
        %mock_grad(inds_to_update) = mock_grad(inds_to_update) + update_vector;
        
        % symmetize update
        inds_to_update = sub2ind([d, d], col_idxs, row_idxs);
        grad_val(inds_to_update) = grad_val(inds_to_update) + rho_k * (n / batch_triang) * update_vector;
        %mock_grad(inds_to_update) = mock_grad(inds_to_update) + update_vector;
    end
    
    % FC STOC GTRAD - grad += rhoK * fc_stoc; grad += 1-rhoK * grad_prev
    grad_val = grad_val(:);
    stoc_gr_f = (total_stoc_obj_fc / batch_fc) * grad_stoc_f(batch_fc, X_lin);
    grad_val = grad_val + rho_k * stoc_gr_f + (1 - rho_k ) * d_k_prev; % add stochastic fc
    
    normX = -1;
    norm_A = -1; 
    is_full = false;
end

function result = lmo_func(X_lin) 
    global trace_bound;
    result = lmo_psd_weight(X_lin, trace_bound);
end

function [grad_val, displacement] = get_equality_constr_grad(X, beta)
    % X needs to be in a matrix shape, nor vectorized
    global d;
    
    displacement = d * trace(X) - sum(sum(X)) - d^2/2;
    grad_val = (1/beta) * (d * eye(d, d) - ones(d, d)) * d * displacement;
end

function [grad_update, update_row_idxs, update_col_idxs, part_feas_normsq] = get_triangle_constr_grad(X, beta, index_triplet)
    % X needs to be in a matrix shape, not vectorized
    i = index_triplet(1);
    j = index_triplet(2);
    k = index_triplet(3);
    t1 = max(X(i, j) + X(j, k) - X(i, k) - X(j, j), 0);
    t2 = max(X(i, k) + X(i, j) - X(j, k) - X(i, i), 0);
    t3 = max(X(i, k) + X(j, k) - X(i, j) - X(k, k), 0);

    grad_update = (1 / beta) * [t1 + t2 - t3,... % for Xij
                                -t1 + t2 + t3, ... % for Xik
                                t1 - t2 + t3, ... % for Xjk
                                -t2/2, ... % for Xii - ./2 because we update these twice when we symmetrize
                                -t1/2, ... % for Xjj - ./2 because we update these twice when we symmetrize
                                -t3/2]; % for Xkk - ./2 because we update these twice when we symmetrize

    update_row_idxs = [i, i, j, i, j, k];
    update_col_idxs = [j, k, k, i, j, k];
    part_feas_normsq = 2 * (t1^2 + t2^2 + t3^2);
end

function [result, num_neg, norm_grad_constr, normA] = var_red_grad_F(v_prev, batch_size, X, X_prev, beta, beta_prev)
    global d;  
    global nchoosek_inds;
    
    % !!!! ONLY CONSTRAINT BATCH MATTERS, AS THE F GRADIENT WILL CANCEL
    % OUT, DUE TO IT BEING CONSTANT
    
    X = reshape(X, [d, d]);
    X_prev = reshape(X_prev, [d, d]);
    
    % EQ CONSTR
    [grad_val, ~] = get_equality_constr_grad(X, beta);
    [grad_val_prev, ~] = get_equality_constr_grad(X_prev, beta_prev);
    
    result = - grad_val_prev + grad_val;
    
    % TRIANGLE_CONSTR
    n = length(nchoosek_inds);
    rand_triang = randperm(n, batch_size);
    for l = 1 : batch_size
        index_triplet = nchoosek_inds(rand_triang(l), :);
        [update_vector, row_idxs, col_idxs, ~] = get_triangle_constr_grad(X, beta, index_triplet);
        [update_vector_prev, ~, ~, ~] = get_triangle_constr_grad(X_prev, beta_prev, index_triplet);
        
        inds_to_update = sub2ind([d, d], row_idxs, col_idxs);
        result(inds_to_update) = result(inds_to_update) + (n / batch_size) * (- update_vector_prev + update_vector);
        
        % symmetize update
        inds_to_update = sub2ind([d, d], col_idxs, row_idxs);
        result(inds_to_update) = result(inds_to_update) +  (n / batch_size) * (- update_vector_prev + update_vector);
    end
    
    result = result(:) + v_prev;
    
    norm_grad_constr = 0;%norm(result - grad_f(X), 2);
    normA = -1; %norm(A_stoch_update, 2);
    num_neg = -1;%length(find(X < 0));
end

function [equality, triangle] = feasibility(X_lin) 
    global d;
    global nchoosek_inds;
    
    X = reshape(X_lin, [d, d]);
    equality = abs(d * trace(X) - sum(sum(X)) - d^2/2);
    
    n = length(nchoosek_inds);
    triangle = 0;
    for l = 1 : n
        inds = nchoosek_inds(l, :);
        i = inds(1);
        j = inds(2);
        k = inds(3);
        triangle = triangle + max(X(i, j) + X(j, k) - X(i, k) - X(j, j), 0)^2 ...
                            + max(X(i, k) + X(i, j) - X(j, k) - X(i, i), 0)^2 ...
                            + max(X(i, k) + X(j, k) - X(i, j) - X(k, k), 0)^2; 
    end
    triangle = sqrt(2 * triangle);
end


