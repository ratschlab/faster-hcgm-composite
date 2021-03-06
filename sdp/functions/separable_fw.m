%% Description
% This implements our H-SAG-CGM (version 2) method for the SDP problems
% (k-means clustering and sparsest cut).
%
% contact: Gideon Dresdner - dgideon@inf.ethz.ch

function [obj_conv_lmo, obj_conv_sfo, constr_conv_lmo, constr_conv_sfo, X_OPT, test_err] = ...
                separable_fw(f, total_stoc_fcs, batches, grad_f, grad_stoc_estimator, feasibility, lmo, X0, beta0, T_eps, test_err_data)

    max_batch = max(batches); % max batch is dictated by symmetry of X. We take 500 random and also their 500 symmetric counterparts.

    obj_conv_lmo = zeros(ceil(T_eps * max_batch/total_stoc_fcs), 1);
    obj_conv_sfo = zeros(ceil(T_eps * max_batch/total_stoc_fcs), 1);
    constr_conv_lmo = zeros(ceil(T_eps * max_batch/total_stoc_fcs), 1);
    constr_conv_sfo =zeros(ceil(T_eps * max_batch/total_stoc_fcs), 1);
    test_err = [];

    d_k = zeros(length(X0), 1);
    X_k = X0;
    constraint_epochs = 1;

    for k = 1 : T_eps
        beta_k = beta0 / (k + 1)^(1/2);
        gamma_k = 2 / (k+1);

        % Record convergence every epoch
        if k == 1 || (max_batch * floor(k/constraint_epochs) >= total_stoc_fcs)
            f_val = f(X_k);
            [feas_eq, feas_ineq] = feasibility(X_k);
            obj_conv_lmo(constraint_epochs) = f_val;
            constr_conv_lmo(constraint_epochs) = feas_eq + feas_ineq;
            obj_conv_sfo(constraint_epochs) = f_val;
            constr_conv_sfo(constraint_epochs) = feas_eq + feas_ineq;
            if mod(k, 10000) == 0 || k == 1
                fprintf("--- constraint_epochs = %d, f = %f, total_feas = %5.4e,  feas_eq = %5.4e, feas_ineq =  %5.4e \n\n\n", constraint_epochs, f_val, feas_eq + feas_ineq, feas_eq, feas_ineq);
            end

            constraint_epochs = constraint_epochs + 1;

        end
        % update
        d_k = grad_stoc_estimator(batches, X_k, beta_k);
        v_t = lmo(d_k);
        X_k = X_k + gamma_k * (v_t - X_k);
    end
    X_OPT = X_k;
end
