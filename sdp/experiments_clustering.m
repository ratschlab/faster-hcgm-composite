%% Description
% This script implements the k-means clustering experiment.
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
addpath('data')
addpath('functions')
addpath('clustering_results')


%% Problem Construction

% use full mnist
% dataset_name = "full_mnist_digits";
% load full_clustering_mnist_digits

% use reduced mnist
dataset_name = "reduced_mnist_digits";
load reduced_clustering_mnist_digits

global C;
global d;
d = Problem.N
C = Problem.C; % Distance Matrix
opt_val = Problem.opt_val
global kappa;
global num_constraints;
global batch_factor_mokhtari;

num_constraints = d^2 + d;
kappa = Problem.k;

global test_err_data;
test_err_data.compute = true;
test_err_data.images = Problem.images;
test_err_data.labels = Problem.labels;
test_err_data.k = kappa;

clearvars Problem

global constant_gradient_f;
constant_gradient_f = C; 
constant_gradient_f = constant_gradient_f(:);

global total_stoc_fcs;
total_stoc_fcs = d + d^2;
batch_factor_mokhtari = 0.05;
fprintf('----- Created matrices');

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
results.test_err = zeros(max_it_sfo, 3);

[V, top_C_eigenval] = eigs(C, 1);
X0 = kappa * (V * V');
trace(X0)
X0 = X0(:);

global grad_X_mult_fact;
grad_X_mult_fact = 1000;
last_epoch_with_part_batch = -1;

global_epoch_count = 1e3;
% global_epoch_count = 1e1;

%% SHCGM - stoc fc
start_exp = tic; 
batch_fc = floor(batch_factor_mokhtari * length(constant_gradient_f));
T_eps = global_epoch_count;
beta0 = 10;
[conv_obj_lmo, conv_obj_sfo, conv_constr_lmo, conv_constr_sfo, X_opt, test_err] = ...
                        shcgm(@f, d^2, batch_fc, @grad_f, @grad_stoc_fc_shcgm, @feasibility, @lmo_func, X0, beta0, T_eps, test_err_data);
fprintf("Time taken = %f seconds \n\n\n", ceil(toc(start_exp))); 
method_names{end+1} = sprintf('shcgm-beta0-%1.5e', beta0);
results.conv_obj_lmo(1 : length(conv_obj_lmo), i) = conv_obj_lmo;
results.conv_obj_sfo(1 : length(conv_obj_sfo), i) = conv_obj_sfo;
results.conv_constr_lmo(1 : length(conv_constr_lmo), i) = conv_constr_lmo;
results.conv_constr_sfo(1 : length(conv_constr_sfo), i) = conv_constr_sfo;
results.last_idx_minib_lmo(:, i) = -1;
results.last_idx_minib_sfo(:, i) = -1;
results.optimal_Xs(:, i) = X_opt;
results.test_err(1:length(test_err), i) = test_err;
i = i + 1;
max_it_lmo = length(conv_obj_lmo);
max_it_sfo = length(conv_obj_sfo);
result_file = sprintf('clustering_results/shcgm_%s_beta0_%.0e_iter_%d_batchfactor_%.3f.mat', datestr(now,'ddmm_HHMMSS'), beta0, T_eps, batch_factor_mokhtari);
save(result_file, 'conv_obj_lmo', 'conv_obj_sfo', 'conv_constr_lmo', 'conv_constr_sfo', 'X_opt', 'opt_val');

%% H-1SFW

betas = (5e-2);
for j = 1 : length(betas)
    start_exp = tic; 
    beta0 = betas(j);
    batch_fc = floor(d^2 * batch_factor_mokhtari);
    batch_eq = floor(d * batch_factor_mokhtari);
    batch_pos = floor(d^2 * batch_factor_mokhtari);
    T_eps = (d^2 / batch_pos) * global_epoch_count;
    %The total_stoc_fcs needs to be d^2 here
    [conv_obj_lmo, conv_obj_sfo, conv_constr_lmo, conv_constr_sfo, X_opt, test_err] = ...
                            mokhtari_as_constr(@f, d^2, [batch_fc, batch_eq, batch_pos], @grad_f, @grad_stoc_full_mokhtari, @feasibility, @lmo_func, X0, beta0, T_eps, test_err_data);
    fprintf("Time taken = %f seconds \n\n\n", ceil(toc(start_exp))); 
    method_names{end+1} = sprintf('mokhtari-as-full-beta0-%5.2e', beta0);
    results.conv_obj_lmo(1 : length(conv_obj_lmo), i) = conv_obj_lmo;
    results.conv_obj_sfo(1 : length(conv_obj_sfo), i) = conv_obj_sfo;
    results.conv_constr_lmo(1 : length(conv_constr_lmo), i) = conv_constr_lmo;
    results.conv_constr_sfo(1 : length(conv_constr_sfo), i) = conv_constr_sfo;
    results.last_idx_minib_lmo(:, i) = -1;
    results.last_idx_minib_sfo(:, i) = -1;
    results.optimal_Xs(:, i) = X_opt;
    results.test_err(1:length(test_err), i) = test_err;
    i = i + 1;
    if max_it_lmo < length(conv_obj_lmo)
        max_it_lmo = length(conv_obj_lmo);
    end
    if max_it_sfo < length(conv_obj_sfo)
        max_it_sfo = length(conv_obj_sfo);
    end
    result_file = sprintf('clustering_results/mokhtari_full_%s_beta0_%.0e_iter_%d_batchfactor_%.3f.mat', datestr(now,'ddmm_HHMMSS'), beta0, T_eps, batch_factor_mokhtari);
    save(result_file, 'conv_obj_lmo', 'conv_obj_sfo', 'conv_constr_lmo', 'conv_constr_sfo', 'X_opt', 'opt_val');
end



%% H-SPIDER-FW 

betas = (6e0);
for j = 1 : length(betas)
    start_exp = tic; 
    beta0 = betas(j);
    T_eps = 10;
    while get_mem_for_sfw(T_eps, d^2, 1) < global_epoch_count
        T_eps = T_eps + 1;
    end
    %   
    % fprintf('--- Starting spiderfw with T_eps = %d, with total_epochs = %d \n', T_eps, get_mem_for_sfw(T_eps, total_stoc_fcs, 10010));
    %T_eps = 20;
    [conv_obj_lmo, conv_obj_sfo, conv_constr_lmo, conv_constr_sfo, X_opt, last_idx_minib_lmo, last_idx_minib_sfo, last_epoch_with_part_batch, test_err] = ...
                    spider_fw(@f, d^2, @grad_parts, @var_red_grad_F, @lmo_func, @feasibility, X0, beta0, T_eps, global_epoch_count, test_err_data);
    fprintf("Time taken = %f seconds \n\n\n", ceil(toc(start_exp))); 
    method_names{end+1} = sprintf('spider-fw-homotopy-beta0-%1.3e', beta0);
    results.conv_obj_lmo(1 : length(conv_obj_lmo), i) = conv_obj_lmo;
    results.conv_obj_sfo(1 : length(conv_obj_sfo), i) = conv_obj_sfo;
    results.conv_constr_lmo(1 : length(conv_constr_lmo), i) = conv_constr_lmo;
    results.conv_constr_sfo(1 : length(conv_constr_sfo), i) = conv_constr_sfo;
    results.last_idx_minib_lmo(1 : length(last_idx_minib_lmo), i) = last_idx_minib_lmo;
    results.last_idx_minib_sfo(1 : length(last_idx_minib_sfo), i) = last_idx_minib_sfo; 
    results.optimal_Xs(:, i) = X_opt;
    results.test_err(1:length(test_err), i) = test_err;
    i = i + 1;
    max_it_lmo = length(conv_obj_lmo);
    max_it_sfo = length(conv_obj_sfo);
    result_file = sprintf('clustering_results/spiderfw_%s_beta0_%.0e_iter_%d.mat', datestr(now,'ddmm_HHMMSS'), beta0, T_eps);
    save(result_file, 'conv_obj_lmo', 'conv_obj_sfo', 'conv_constr_lmo', 'conv_constr_sfo', ...
                                                'last_idx_minib_lmo', 'last_idx_minib_sfo', 'X_opt', 'opt_val');
end


%% Stochastic Separable FW
betas = (7);
global alpha_k;
global gamma_k;
for j = 1 : length(betas)
    start_exp = tic;
    beta0 = betas(j);
    batch_fc = floor(d^2 * batch_factor_mokhtari);
    batch_eq = floor(d * batch_factor_mokhtari);
    batch_pos = floor(d^2 * batch_factor_mokhtari);
    T_eps = (d^2 / batch_pos) * global_epoch_count;
    
    alpha_k = zeros(size(X0));
    gamma_k = zeros(1, d^2);
    
    [conv_obj_lmo, conv_obj_sfo, conv_constr_lmo, conv_constr_sfo, X_opt, test_err] = ...
        separable_fw(@f, d^2, [batch_fc, batch_eq, batch_pos], @grad_f, @grad_separable, @feasibility, @lmo_func, X0, beta0, T_eps, test_err_data);
    fprintf("Time taken = %f seconds \n\n\n", ceil(toc(start_exp)));
    method_names{end+1} = sprintf('stochastic-separable-as-full-beta0-%5.2e', beta0);
    results.conv_obj_lmo(1 : length(conv_obj_lmo), i) = conv_obj_lmo;
    results.conv_obj_sfo(1 : length(conv_obj_sfo), i) = conv_obj_sfo;
    results.conv_constr_lmo(1 : length(conv_constr_lmo), i) = conv_constr_lmo;
    results.conv_constr_sfo(1 : length(conv_constr_sfo), i) = conv_constr_sfo;
    results.last_idx_minib_lmo(:, i) = -1;
    results.last_idx_minib_sfo(:, i) = -1;
    results.optimal_Xs(:, i) = X_opt;
    results.test_err(1:length(test_err), i) = test_err;
    i = i + 1;
    if max_it_lmo < length(conv_obj_lmo)
        max_it_lmo = length(conv_obj_lmo);
    end
    if max_it_sfo < length(conv_obj_sfo)
        max_it_sfo = length(conv_obj_sfo);
    end
    result_file = sprintf('clustering_results/separable_fw_%s_beta0_%.0e_iter_%d_batchfactor_%.3f.mat', datestr(now,'ddmm_HHMMSS'), beta0, T_eps, batch_factor_mokhtari);
    save(result_file, 'conv_obj_lmo', 'conv_obj_sfo', 'conv_constr_lmo', 'conv_constr_sfo', 'X_opt', 'opt_val');
end

results.method_names = method_names;

save(sprintf('clustering_results/iter_%d_batchfactor_%.3f_dataset=%s.mat', T_eps, batch_factor_mokhtari, dataset_name), ...
    '-struct', 'results');

%% Ploting
num_methods = length(method_names);
method_names = ["SHCGM", "H-1SFW", "H-SPIDER-FW", "H-SAG-CGM"];
plot_convergence(...
    method_names(1:num_methods), results.conv_obj_lmo(1: max_it_lmo, 1:num_methods), ...
    results.conv_obj_lmo(1: max_it_lmo, 1:num_methods), results.conv_obj_sfo(1 : max_it_sfo, 1:num_methods),...
    results.conv_constr_lmo(1: max_it_lmo, 1:num_methods), results.conv_constr_sfo(1 : max_it_sfo, 1:num_methods), ...
    results.optimal_Xs(:, 1:num_methods), [], results.last_idx_minib_lmo(:, :), results.last_idx_minib_sfo(:, :), results.test_err(1: max_it_lmo, 1:num_methods), ...
    last_epoch_with_part_batch, opt_val, total_stoc_fcs);

function result = f(X_lin)
    global C;
    result = dot(C(:), X_lin);
end

function result = grad_f(~)
    global constant_gradient_f;
    result = constant_gradient_f;
end

function [result, idx] = grad_stoc_f(batch_size, ~)
    global constant_gradient_f;
    rand_ind = randperm(length(constant_gradient_f), batch_size);
    
    result = zeros(length(constant_gradient_f), 1);
    result(rand_ind) = constant_gradient_f(rand_ind);
    
    idx = rand_ind;
end

function [d_k, grad_constr, useless2, useless3] = grad_stoc_fc_shcgm(batch_fc, X_lin, beta, rho_k, d_k_prev)
    global d;
    
    AX_minus_b = A1(X_lin) - 1;
    min_X_0 = min(X_lin, 0);
    
    %%%%% I NEED TO TAKE SYMMETRIC F HERE
    d_k = (1 - rho_k ) * d_k_prev + rho_k * ( d^2 / batch_fc) * grad_stoc_f(batch_fc, X_lin);
    grad_constr = d * (1/beta) * min_X_0;
    grad_constr = reshape(grad_constr, [d, d]);
    grad_constr = grad_constr + (1/beta) * AX_minus_b;
    grad_constr = grad_constr + (1/beta) * transpose(AX_minus_b);
    grad_constr = grad_constr(:);
    
    useless2 = 0;
    useless3 = false;
end

function [result, normX, norm_A, is_full] = grad_stoc_full_mokhtari(batches, X_lin, beta, rho_k, d_k_prev)
    % Checked that if I turn these parameters into the ones corresponding
    % to the deterministic part, it will yield same result as homotopy cgm.
    global d;
    batch_fc = batches(1);
    batch_eq = batches(2);
    batch_pos = batches(3);
    
    % 2. ACTIVE CONSTRAINT SELECTION: Update positivity constraints. 
    neg_indices = find(X_lin < 0);
    num_neg = length(neg_indices);
    batch_pos = min(batch_pos, num_neg);
    rand_ind_pos = neg_indices(randperm(num_neg, batch_pos));
    
    % 2. Get random indices and their symmetric counterparts
    %rand_ind_pos = randperm(d^2, batch_pos); 
    
    result = (1 - rho_k) * d_k_prev;
    result = result + rho_k * (d^2 / batch_fc) * grad_stoc_f(batch_fc, X_lin);
    X_stoch = d * (num_neg / batch_pos) * (1/beta) * X_lin(rand_ind_pos); % I don't need to do min, because they are already negative indices
    %X_stoch = d * (d^2 / batch_pos) * (1/beta) * min(X_lin(rand_ind_pos), 0); % multiply by something here. the 10 is for baalncing out the weight of the updates
    result(rand_ind_pos) = result(rand_ind_pos) + rho_k * X_stoch;
    
    
    % 3. Update the Equality constraints
    result = reshape(result, [d,d]);
    rand_ind_eq = randperm(d, batch_eq);
    Astoc_X_minus_b = (d / batch_eq) * (1/beta) * (A1stoc(rand_ind_eq, X_lin) - 1);
    result(rand_ind_eq, :) = result(rand_ind_eq, :) + rho_k * Astoc_X_minus_b; % this is equivalent to applying the transpose operator for lines and for columns
    result(:, rand_ind_eq) = result(:, rand_ind_eq) + rho_k * transpose(Astoc_X_minus_b);
    
    %fprintf('norm_A_batch = %f, norm_X_batch = %f\n\n', norm(A_batch), norm(X_batch));
    result = result(:);
    normX = norm(X_stoch * beta, 2);
    norm_A = norm(Astoc_X_minus_b * beta, 2);
    is_full = length(find(X_lin<0));
end

function [stoch_grad_obj, grad_pos_constr, grad_eq_constr] = grad_parts(batches, X_lin)
    global d;
    batch_fc = batches(1);
    batch_eq = batches(2);
    
    %% Stochastic Objective
    stoch_grad_obj = zeros([d,d]);
    [f_sg, f_sg_idx] = grad_stoc_f(batch_fc, X_lin);
    stoch_grad_obj(f_sg_idx) = f_sg(f_sg_idx);
    
    %% Deterministic Positivity Constraints
    min_X_0 = min(X_lin, 0);
    grad_pos_constr = min_X_0;
    
    %% Stochastic Equality Constraints
    grad_eq_constr = zeros([d,d]);
    rand_ind_eq = randperm(d, batch_eq);
    Astoc_X_minus_b = (A1stoc(rand_ind_eq, X_lin) - 1);
    grad_eq_constr(rand_ind_eq, :) = repmat(Astoc_X_minus_b, [1,d]);
    grad_eq_constr(:, rand_ind_eq) = transpose(repmat(Astoc_X_minus_b, [1,d]));
    
    stoch_grad_obj = (d^2/batch_fc)*stoch_grad_obj;
    grad_pos_constr = d*grad_pos_constr;
    grad_eq_constr = (d/batch_eq)*grad_eq_constr;
end

function [d_k, unused1, unused2] = grad_separable(batches, X_lin, beta)
    global d;
    batch_fc = batches(1);
    batch_eq = batches(2);
%     batch_pos = batches(3);
    
    %% Stochastic Objective
    [f_sg, f_sg_idx] = grad_stoc_f(batch_fc, X_lin);
    global alpha_k;
    alpha_k(f_sg_idx) = f_sg(f_sg_idx);
    
    %% Deterministic Positivity Constraints
    min_X_0 = min(X_lin, 0);
    grad_constr = min_X_0;
    
    %% Stochastic Equality Constraints

    global gamma_k;
    grad_eq_constraints = reshape(gamma_k, [d,d]);
    rand_ind_eq = randperm(d, batch_eq);
    Astoc_X_minus_b = (A1stoc(rand_ind_eq, X_lin) - 1);
    grad_eq_constraints(rand_ind_eq, :) = repmat(Astoc_X_minus_b, [1,d]);
    grad_eq_constraints(:, rand_ind_eq) = transpose(repmat(Astoc_X_minus_b, [1,d]));
    gamma_k = grad_eq_constraints(:);

    d_k = (d^2/batch_fc)*alpha_k + d*(1/beta)*grad_constr + (d/batch_eq)*(1/beta)*gamma_k;
    
    unused1 = -1;
    unused2 = -1;
end

function result = lmo_func(X_lin) 
    global kappa;
    result = lmo_psd_weight(X_lin, kappa);
end

function result = A1(X_lin)
    % will return a column vector of size d, containing the column sums sums
    global d;

    result = transpose(sum(reshape(X_lin, [d, d]), 1));
end

function result = A1stoc(rand_indices, X_lin)
    % will return a column vector of size d, containing the column sums sums
    global d;
    
    tmp = reshape(X_lin, [d, d]);
    result = transpose(sum(tmp(:, rand_indices), 1));
end

function [result, num_neg, norm_grad_constr, normA] = var_red_grad_F(v_prev, batch_size, X, X_prev, beta, beta_prev)
    global d;  
    assert(batch_size >= 1);

    batch_eq = floor(sqrt(batch_size));
    batch_eq = min(max(batch_eq, 1), d);
    
    neg_indices = find(X < 0);
    num_neg = length(neg_indices);
    batch_pos = min(batch_size, num_neg);

    
    result = v_prev;
    if batch_pos ~= 0
        rand_ind_pos = neg_indices(randperm(num_neg, batch_pos));
        X_stoch_update = -(1/beta_prev) * min(X_prev(rand_ind_pos), 0) + (1/beta) * min(X(rand_ind_pos), 0);
        result(rand_ind_pos) = result(rand_ind_pos) + (d * num_neg / batch_pos) * X_stoch_update;
    else
        rand_idx = randperm(d^2, batch_size);
        X_stoch_update = -(1/beta_prev) * min(X_prev(rand_idx), 0) + (1/beta) * min(X(rand_idx), 0);
        result(rand_idx) = result(rand_idx) + (d * d^2 / batch_size) * X_stoch_update;
    end

    % update equality
    rand_ind_eq = randperm(d, batch_eq); 
    result = reshape(result, [d, d]);
    AXprev_minus_b = A1stoc(rand_ind_eq, X_prev) - 1;
    AX_minus_b = A1stoc(rand_ind_eq, X) - 1;
    A_stoch_update = -(1/beta_prev) * AXprev_minus_b + (1/beta) * AX_minus_b;
    result(rand_ind_eq, :) = result(rand_ind_eq, :) + (d/batch_eq) * A_stoch_update;
	result(:, rand_ind_eq) = result(:, rand_ind_eq) + (d/batch_eq) * transpose(A_stoch_update);

	result = result(:);
    norm_grad_constr = 0;%norm(result - grad_f(X), 2);
    normA = -1; %norm(A_stoch_update, 2);
    num_neg = -1;%length(find(X < 0));
end

function [equality, positivity] = feasibility(X_lin) 
    global d;
    equality = norm(A1(X_lin) - 1, 2)/sqrt(d);
    positivity = norm(min(X_lin, 0), 2);
end
