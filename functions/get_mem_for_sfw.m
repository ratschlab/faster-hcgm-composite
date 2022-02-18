function constraint_epochs = get_mem_for_sfw(T_eps, total_stoc_fcs, constant)  
    total_sfos = 0;
    for t = 1 : T_eps
        % account for full gradient
        total_sfos = total_sfos + total_stoc_fcs;
        
        % count for inner iteration
        K_t = 2^(t - 1);
        batch_size = min(total_stoc_fcs, K_t * constant);
        total_sfos =  total_sfos + (K_t/4 - 1) * batch_size;
    end
    constraint_epochs = floor(total_sfos / total_stoc_fcs);
end