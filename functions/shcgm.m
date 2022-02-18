function [obj_conv_lmo, obj_conv_sfo, constr_conv_lmo, constr_conv_sfo, X_OPT, test_err] = ...
                shcgm(f, num_constraints, batch_fc, grad_f, grad_stoc_estimator, feasibility, lmo, X0, beta0, T_eps, test_err_data)

    obj_conv_lmo = zeros(T_eps, 1);
    obj_conv_sfo = zeros(T_eps, 1);
    constr_conv_lmo = zeros(T_eps, 1);
    constr_conv_sfo =zeros(T_eps, 1);
    test_err = [];
    
    d_k = zeros(length(X0), 1);
    X_k = X0;
    constraint_epochs = 1;
    
    for k = 1 : T_eps
        gamma_k = 9/(k+8);
        beta_k = beta0/sqrt(k+8);
        rho_k = 4/(k+7)^(2/3);
        
        % Record convergence every iteration, as th constraints are being
        % processed in full
        f_val = f(X_k);
        [feas_eq, feas_ineq] = feasibility(X_k);
        obj_conv_lmo(constraint_epochs) = f_val;
        constr_conv_lmo(constraint_epochs) = feas_eq + feas_ineq; 
        obj_conv_sfo(constraint_epochs) = f_val;
        constr_conv_sfo(constraint_epochs) = feas_eq + feas_ineq; 
        
        constraint_epochs = constraint_epochs + 1;
        
        if k == 1 || mod(k, 1000) == 0
            fprintf("---k = %d/ %d, f = %f , total_feas = %3.5e, feas_eq = %3.5e, feas_ineq = %3.5e \n", k, T_eps, f_val, feas_eq + feas_ineq, feas_eq, feas_ineq);
        end
        
        % update
        [d_k, full_grad_constr, ~, ~] = grad_stoc_estimator(batch_fc, X_k, beta_k, rho_k, d_k);
        v_t = lmo(d_k + full_grad_constr);
        X_k = X_k + gamma_k * (v_t - X_k);
    end
    
    X_OPT = X_k;
end
