function plot_convergence(method_names, vec_full_conv_obj, vec_obj_conv_lmo, vec_obj_conv_sfo, vec_constr_conv_lmo, vec_constr_conv_sfo,...
    optimal_Xs, X_true, vec_last_idx_minib_lmo, vec_last_idx_minib_sfo, test_err, last_epoch_with_part_batch,  f_opt, total_stoc_fcs)
    
    num_methods = length(method_names);
    figures_folder = 'paper_plots/';
        
    vec_obj_conv_lmo = abs((vec_obj_conv_lmo - f_opt)/abs(f_opt));
    vec_obj_conv_sfo = abs((vec_obj_conv_sfo - f_opt)/abs(f_opt));
    
    vec_constr_conv_lmo = vec_constr_conv_lmo; 
    vec_constr_conv_sfo = vec_constr_conv_sfo;
    
    inferior_y_lim = 1e-6;
    superior_x_lim_lmo = length(vec_obj_conv_lmo);
    superior_x_lim_sfo = length(vec_obj_conv_sfo);
    
    fntsize = 20;  
    colors = {'-k', '-b', '-r', '-g'};
%     matplotlib_colors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...
%         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'};
    
% https://github.com/mwaskom/seaborn/blob/4fe84d4ff25276f0ef4ea2a18d0a31741547cb0f/seaborn/palettes.py#L40
    matplotlib_colors = {"#DE8F05", "#029E73", "#D55E00","#0173B2", "#CC78BC", ...
                "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"};
    
    target_method = 'separable';

    figure('Name', 'SFO convergence objective')
        for i = 1 : num_methods
            linewdth = 2;
%             loglog(vec_obj_conv_sfo(:, i), colors{i}, 'DisplayName', method_names{i}, 'LineWidth', linewdth);%             loglog(vec_obj_conv_sfo(:, i), colors{i}, 'DisplayName', method_names{i}, 'LineWidth', linewdth);
            
            if strfind(method_names{i}, target_method)
%                 color = 'k';
                color = matplotlib_colors{i};
            else
                color = matplotlib_colors{i};
            end
                
            loglog(vec_obj_conv_sfo(:, i), 'Color', color, 'DisplayName', method_names{i}, 'LineWidth', linewdth);
            
            hold on;
%             if strfind(method_names{i}, 'spider-fw')
%                 xline(last_epoch_with_part_batch, '-b', 'LineWidth', linewdth, 'HandleVisibility','off');
% %                 for j = 1 : length(vec_last_idx_minib_lmo(:, i))
% %                     if vec_last_idx_minib_lmo(j, i) == 0
% %                         break;
% %                     end
% %                     xline(vec_last_idx_minib_sfo(j, i), '--r', 'HandleVisibility','off'); 
% %                 end
%             end 
        end
        %ylim([inferior_y_lim inf]);
        %xline(last_idx_minib_sfo, '--r', 'DisplayName', 'last minibatch');
        %xlim([-inf superior_x_lim_sfo]);
        grid on;
        legend('Location', 'southwest', 'FontSize',fntsize);
        %title('Objective convergence');
%         ylabel("$\frac{\left | f(X) - f^* \right |}{f^*}$", 'interpreter','latex', 'FontSize',fntsize)
        ylabel("Relative Suboptimality", 'interpreter','latex', 'FontSize',fntsize)
        xlabel('Constraint epochs','interpreter','latex', 'FontSize',fntsize);
        set(gca,'FontSize',fntsize)
        hold off;

    figure('Name', 'SFO convergence constraints')

        for i = 1 : num_methods
            linewdth = 2;
            
            if strfind(method_names{i}, target_method)
%                 color = 'k';
                color = matplotlib_colors{i};
            else
                color = matplotlib_colors{i};
            end
            
            loglog(vec_constr_conv_sfo(:, i), 'Color', color, 'DisplayName', method_names{i},'LineWidth', linewdth);
            hold on;
            if strfind(method_names{i}, 'spider-fw')
                xline(last_epoch_with_part_batch, '-b', 'LineWidth', linewdth, 'HandleVisibility','off'); 
%                 for j = 1 : length(vec_last_idx_minib_lmo(:, i))
%                     if vec_last_idx_minib_lmo(j, i) == 0
%                         break;
%                     end
%                     xline(vec_last_idx_minib_sfo(j, i), '--r', 'HandleVisibility','off'); 
%                 end
            end
        end
        %ylim([inferior_y_lim inf]);
        %xline(last_idx_minib_sfo, '--r', 'DisplayName', 'last minibatch');
        %xlim([-inf superior_x_lim_sfo]);
        grid on;
        legend('Location', 'southwest', 'FontSize', fntsize);
%         ylabel("$\| AX - b \|$", 'interpreter','latex', 'FontSize',fntsize)
        ylabel("Distance to Feasibility", 'interpreter','latex', 'FontSize',fntsize)
        xlabel('Constraint epochs','interpreter','latex', 'FontSize',fntsize);
        set(gca,'FontSize',fntsize)
        %title('Constraint convergence');
        
% 	figure('Name', 'Misclassification Rate')
% 
%         for i = 1 : num_methods
%             linewdth = 2;
% 
%             semilogx(test_err(:, i),  'DisplayName', method_names{i},'LineWidth', linewdth);
%             hold on;
%             if strfind(method_names{i}, 'spider-fw')
%                 xline(last_epoch_with_part_batch, '-b', 'LineWidth', linewdth, 'HandleVisibility','off'); 
% %                 for j = 1 : length(vec_last_idx_minib_lmo(:, i))
% %                     if vec_last_idx_minib_lmo(j, i) == 0
% %                         break;
% %                     end
% %                     xline(vec_last_idx_minib_sfo(j, i), '--r', 'HandleVisibility','off'); 
% %                 end
%             end
%         end
%         %ylim([inferior_y_lim inf]);
%         %xline(last_idx_minib_sfo, '--r', 'DisplayName', 'last minibatch');
%         %xlim([-inf superior_x_lim_sfo]);
%         grid on;
%         legend('Location', 'southwest', 'FontSize', fntsize);
%         ylabel("Misclassification rate", 'interpreter','latex', 'FontSize',fntsize)
%         xlabel('Constraint epochs','interpreter','latex', 'FontSize',fntsize);
%         set(gca,'FontSize',fntsize)
%         %title('Constraint convergence');

           
    %saveas(fig, strcat(figures_folder, strjoin(method_names, '|'), int2str(length(b)), '-', int2str(length(X_true))) , 'fig');
% %         
%        d = sqrt(length(optimal_Xs(:, 1)));
%        figure('Name', 'Solutions')
%             n = ceil(length(num_methods)/2);
%             n = n + 1;
%             for i = 1 : num_methods
%                 subplot(n, 2, i);
%                 imagesc(reshape(optimal_Xs(:, i), [d, d]));
%                 title(strcat(method_names(i), ' Sol'));
%             end
% 
%             subplot(n, 2, num_methods + 1);
%             imagesc(X_true);
%             title('True Sol');
end

