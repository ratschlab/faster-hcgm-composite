close all

% Figure size and name
hfig = figure(...
    'Position',[100,100,1300,650],...
    'name','SDP-experiments',...
    'numbertitle','off');

% Definition of error measures
plot_obj_conv = @(METHOD) loglog (abs(METHOD.conv_obj_lmo - METHOD.opt_val) );
plot_constr_conv = @(METHOD) loglog (METHOD.conv_constr_sfo);

%% Clustering

% load data
HSAGCGM = load("clustering_results/separable_fw_1010_222956_beta0_7e+00_iter_20000_batchfactor_0.050.mat");
HSPIDERFW = load("clustering_results/spiderfw_1010_221846_beta0_6e+00_iter_17.mat");
H1SFW = load("clustering_results/mokhtari_full_1010_214809_beta0_5e-02_iter_20000_batchfactor_0.050.mat");
SHCGM = load("clustering_results/shcgm_1010_213318_beta0_1e+01_iter_1000_batchfactor_0.050.mat");
names = {"H-SAG-CGM", "H-SPIDER-FW", "H-1SFW", "SHCGM"};

% Objective suboptimality
subplot(241)
plot_obj_conv(HSAGCGM); hold on;
plot_obj_conv(HSPIDERFW);
plot_obj_conv(H1SFW);
plot_obj_conv(SHCGM); hold off;
ylabel("objective residual",'Interpreter','latex','FontSize',13);
xlabel('epoch','Interpreter','latex','FontSize',13);
title({'\textbf{k-means clustering}','\texttt{MNIST} (preprocessed)'},'Interpreter','latex');

% Infeasibility error
subplot(242)
plot_constr_conv(HSAGCGM); hold on;
plot_constr_conv(HSPIDERFW);
plot_constr_conv(H1SFW);
plot_constr_conv(SHCGM); hold off;
ylabel("infeasibility error", 'interpreter','latex', 'FontSize',13);
xlabel('epoch','interpreter','latex', 'FontSize',13);
title({'\textbf{k-means clustering}','\texttt{MNIST} (preprocessed)'},'Interpreter','latex');

%% Sparsest-cut
% 25mammalia-primate-association-13

% load data
HSAGCGM = load("sp_cut_results/25mammalia-primate-association-13/separable_1010_162249_beta0_1e+02_iter_20000_batchfactor_0.050.mat");
HSPIDERFW = load("sp_cut_results/25mammalia-primate-association-13/spiderfw_1010_162206_beta0_1e+01_iter_13.mat");
H1SFW = load("sp_cut_results/25mammalia-primate-association-13/mokhtari_full_1010_162148_beta0_1e-02_iter_20000_batchfactor_0.050.mat");
SHCGM = load("sp_cut_results/25mammalia-primate-association-13/shcgm_1010_162125_beta0_1e+02_iter_1000_batchfactor_0.050.mat");

% Objective suboptimality
subplot(243)
plot_obj_conv(HSAGCGM); hold on;
plot_obj_conv(HSPIDERFW);
plot_obj_conv(H1SFW);
plot_obj_conv(SHCGM); hold off;
ylabel("objective residual",'Interpreter','latex','FontSize',13);
xlabel('epoch','Interpreter','latex','FontSize',13);
title({'\textbf{sparsest cut}','\texttt{25mammalia-primate-association-13}'},'Interpreter','latex');

% Infeasibility error
subplot(244)
plot_constr_conv(HSAGCGM); hold on;
plot_constr_conv(HSPIDERFW);
plot_constr_conv(H1SFW);
plot_constr_conv(SHCGM); hold off;
ylabel("infeasibility error", 'interpreter','latex', 'FontSize',13);
xlabel('epoch','interpreter','latex', 'FontSize',13);
title({'\textbf{sparsest cut}','\texttt{25mammalia-primate-association-13}'},'Interpreter','latex');


%% Sparsest-cut
% 55n-insecta-ant-colony1-day37

% load data
HSAGCGM = load("sp_cut_results/55n-insecta-ant-colony1-day37/separable_1010_173006_beta0_1e+02_iter_2.001144e+04_batchfactor_0.050.mat");
HSPIDERFW = load("sp_cut_results/55n-insecta-ant-colony1-day37/spiderfw_1010_172328_beta0_1e+01_iter_15.mat");
H1SFW = load("sp_cut_results/55n-insecta-ant-colony1-day37/mokhtari_full_1010_171915_beta0_1e-02_iter_2.001144e+04_batchfactor_0.050.mat");
SHCGM = load("sp_cut_results/55n-insecta-ant-colony1-day37/shcgm_1010_171501_beta0_1e+02_iter_1000_batchfactor_0.050.mat");

% Objective suboptimality
subplot(245)
plot_obj_conv(HSAGCGM); hold on;
plot_obj_conv(HSPIDERFW);
plot_obj_conv(H1SFW);
plot_obj_conv(SHCGM); hold off;
ylabel("objective residual",'Interpreter','latex','FontSize',13);
xlabel('epoch','Interpreter','latex','FontSize',13);
title({'\textbf{sparsest cut}','\texttt{55n-insecta-ant-colony1-day37}'},'Interpreter','latex');

hlegend = legend(names); % I put legend here because there is a lot of space here
hlegend.Interpreter = 'latex';
hlegend.FontSize = 14;
hlegend.Location = 'SouthWest';

% Infeasibility error
subplot(246)
plot_constr_conv(HSAGCGM); hold on;
plot_constr_conv(HSPIDERFW);
plot_constr_conv(H1SFW);
plot_constr_conv(SHCGM); hold off;
ylabel("infeasibility error", 'interpreter','latex', 'FontSize',13);
xlabel('epoch','interpreter','latex', 'FontSize',13);
title({'\textbf{sparsest cut}','\texttt{55n-insecta-ant-colony1-day37}'},'Interpreter','latex');

%% Sparsest-cut
% 102n-insecta-ant-colony4-day10

% load data
HSAGCGM = load("sp_cut_results/102n-insecta-ant-colony4-day10/separable_1010_192447_beta0_1e+02_iter_20000_batchfactor_0.050.mat");
HSPIDERFW = load("sp_cut_results/102n-insecta-ant-colony4-day10/spiderfw_1010_185145_beta0_1e+01_iter_16.mat");
H1SFW = load("sp_cut_results/102n-insecta-ant-colony4-day10/mokhtari_full_1010_183036_beta0_1e-02_iter_20000_batchfactor_0.050.mat");
SHCGM = load("sp_cut_results/102n-insecta-ant-colony4-day10/shcgm_1010_181021_beta0_1e+02_iter_1000_batchfactor_0.050.mat");

% Objective suboptimality
subplot(247)
plot_obj_conv(HSAGCGM); hold on;
plot_obj_conv(HSPIDERFW);
plot_obj_conv(H1SFW);
plot_obj_conv(SHCGM); hold off;
ylabel("objective residual",'Interpreter','latex','FontSize',13);
xlabel('epoch','Interpreter','latex','FontSize',13);
title({'\textbf{sparsest cut}','\texttt{102n-insecta-ant-colony4-day10}'},'Interpreter','latex');

% Infeasibility error
subplot(248)
plot_constr_conv(HSAGCGM); hold on;
plot_constr_conv(HSPIDERFW);
plot_constr_conv(H1SFW);
plot_constr_conv(SHCGM); hold off;
ylabel("infeasibility error", 'interpreter','latex', 'FontSize',13);
xlabel('epoch','interpreter','latex', 'FontSize',13);
title({'\textbf{sparsest cut}','\texttt{102n-insecta-ant-colony4-day10}'},'Interpreter','latex');

%% Display properties for all subplots
for t = 1:8
    subplot(2,4,t)
    
    ax = gca;
    set(findall(ax, 'Type', 'line'),'LineWidth',2);
    ax.FontSize = 13;
    ax.TickLabelInterpreter = 'latex';
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.TickLength = [0.02 0.02];
    ax.Box = 'on';
    
    htitle=get(gca,'title');
    htitle.FontSize = 13;
    
    ax.YTick = 10.^(-100:1:100);
    ax.XTick = 10.^(-100:1:100);
    
    grid on, grid minor, grid minor;
end

return;

%% Save Figures
% This saves pdf's perfectly cropped etc!

% First download altmany-export_fig-d966721 and add to path
addpath('/Users/alp/Documents/MATLAB/altmany-export_fig-d966721/')

% Determine and create the path to save the figures
figPath = ['figs/',datestr(now,30),'/'];
mkdir(figPath);

figHandles = findall(groot, 'Type', 'figure');
for rr = 1:length(figHandles)
    sName = [figPath,figHandles(rr).Name];
    % savefig(figHandles(rr), sName)
    set(0, 'CurrentFigure', figHandles(rr))
    figHandles(rr).Color = [1,1,1];
    if exist('export_fig','file')
        export_fig(sName,'-pdf','-dCompatibilityLevel=1.4')
    end
end
