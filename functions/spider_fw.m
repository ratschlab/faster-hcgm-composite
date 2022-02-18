function [obj_lmo, obj_sfo, constr_lmo, constr_sfo, X_OPT, last_index_with_partial_batch, last_index_with_partial_batch_sfo, last_epoch_with_part_batch, test_err] = ...
                            spider_fw(f, total_stoc_fcs, grad_F, var_red_grad, lmo, feasibility, X0, beta0, T_eps, max_epochs, test_err_data)
    
    batch_factor = 1;
    num_proposed_epochs = min(get_mem_for_sfw(T_eps, total_stoc_fcs, batch_factor), max_epochs);
    epochs_to_be_done = min(max_epochs, num_proposed_epochs);

    obj_lmo = zeros(max_epochs, 1);
    constr_lmo = zeros(max_epochs, 1);
    obj_sfo = zeros(max_epochs, 1);
    constr_sfo = zeros(max_epochs, 1);
    test_err = [];
    
    last_index_set = false;
    last_index_with_partial_batch = zeros(T_eps, 1);
    last_index_with_partial_batch_sfo = zeros(T_eps, 1);
    
    eta_rate = @(K_t, k)  (2 / (K_t + k));
    beta_rate = @(K_t, k) (beta0 / sqrt(K_t + k));

    X_bar = X0;
    constraint_epochs = 1;
    current_sfo_cnt = 0;
    last_epoch_with_part_batch = -1;
    for t = 1 : T_eps
        K_t = 2^(t - 1);
        batch_size = min(total_stoc_fcs, K_t * batch_factor);
        
        eta_t_prev = eta_rate(K_t, 1);
        beta_t_prev = beta_rate(K_t, 1);
        X_prev = X_bar; 
        
%         [v_prev, feas_eq, feas_ineq, ~] = grad_F(X_prev, beta_t_prev); % full grad

        arbitrary_stoch_bs = max(1,min(250, floor(2*K_t*(K_t+1)/beta0)));
        [f_grad, pos_grad, eq_grad] = grad_F([arbitrary_stoch_bs,arbitrary_stoch_bs], X_prev);
        v_prev = f_grad(:) + pos_grad + 1/beta_t_prev*eq_grad(:);
        
        X_k = X_prev + eta_t_prev * (lmo(v_prev) - X_prev);
        
        % 1. Record convergence: constraint epoch was performed (full grad) => record
%         f_val = f(X_prev);
%         obj_lmo(constraint_epochs) = f_val;
%         constr_lmo(constraint_epochs) = feas_eq + feas_ineq;
%         obj_sfo(constraint_epochs) = f_val;
%         constr_sfo(constraint_epochs) = feas_eq + feas_ineq;
%         last_index_with_partial_batch(t) = constraint_epochs;
%         last_index_with_partial_batch_sfo(t) = constraint_epochs;
        
%         fprintf("--- epoch_fullG %d/%d, f = %f, tot_feas = %5.4e, feas_eq = %5.4e, feas_ineq =  %5.4e, beta_k = %f, eta_k = %f, batch_sz = %d, num_neg_X = %d,  \n\n\n",...
%                     constraint_epochs, epochs_to_be_done, f_val, feas_eq + feas_ineq, feas_eq, feas_ineq, beta_t_prev, eta_t_prev, total_stoc_fcs, -1); 
%         
        % 2. Update the counters. The current sfo counter does not need to
        % be changed, here, whatever it has will be transported to the
        % stochastic stuff.
        if constraint_epochs == max_epochs
            X_bar = X_k;
            break;
        end
        
%         constraint_epochs = constraint_epochs + 1;
        current_sfo_cnt = current_sfo_cnt + arbitrary_stoch_bs;
        
        for k = 2 : K_t
            eta_t_k = eta_rate(K_t, k);
            beta_t_k = beta_rate(K_t, k);
            
            [v_k, ~, ~, ~] = ...
                        var_red_grad(v_prev, batch_size, X_k, X_prev, beta_t_k, beta_t_prev);
            
            X_k_next = X_k + eta_t_k * (lmo(v_k) - X_k);

            beta_t_prev = beta_t_k;
            X_prev = X_k;
            X_k = X_k_next;
            v_prev = v_k;
            
            if current_sfo_cnt + batch_size >= total_stoc_fcs
                f_val = f(X_prev); 
                [feas_eq, feas_ineq] = feasibility(X_prev); 
                obj_lmo(constraint_epochs) = f_val;
                constr_lmo(constraint_epochs) = feas_eq + feas_ineq;
                obj_sfo(constraint_epochs) = f_val;
                constr_sfo(constraint_epochs) = feas_eq + feas_ineq;
                
                % 2. Reset the counter
                current_sfo_cnt = (current_sfo_cnt + batch_size) - total_stoc_fcs;
                
                if mod(constraint_epochs, 50) == 0
                    fprintf("--- epoch_partG = %d/%d, f = %f, tot_feas = %5.4e,  feas_eq = %5.4e, feas_ineq =  %5.4e, beta_k = %1.7e, eta_k = %1.7e, batch_sz = %d, num_neg_X = %d \n\n\n",...
                            constraint_epochs, epochs_to_be_done, f_val, feas_eq + feas_ineq, feas_eq, feas_ineq, beta_t_k, eta_t_k, batch_size, -1);
                end
                
                if constraint_epochs == max_epochs
                    break;
                end
                constraint_epochs = constraint_epochs + 1;
            else
                current_sfo_cnt = current_sfo_cnt + batch_size;
            end    
        end
        X_bar = X_k; 
        
        %fprintf('\n============== end outer iter %d / %d. Took %f seconds. =================\n\n', t, T_eps, toc(tstart_epoch));
        if and(last_index_set == false, 2 * K_t > total_stoc_fcs)
            last_index_set = true;
            last_epoch_with_part_batch = constraint_epochs;
        end
    end
    
    obj_lmo(constraint_epochs + 1 : end) = [];
    constr_lmo(constraint_epochs + 1 : end) = [];
    obj_sfo(constraint_epochs + 1: end) = [];
    constr_sfo(constraint_epochs + 1 : end) = [];
    
    X_OPT = X_bar;
end