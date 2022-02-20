%% Description
% This script generates the plots for Figure 4 in [Ref].
% You need to run the test file TESTMOVIELENS100k first, before this 
% script.
%
% [Ref] Locatello, F., Yurtsever, A., Fercoq, O., Cevher, V.
% "Stochastic Conditional Gradient Method for Composite Convex Minimization"
% Advances in Neural Information Processing Systems 32 (NeurIPS 2019).
%
% contact: Alp Yurtsever - alp.yurtsever@epfl.ch
% Modified from the code for [LY+19] 

%% Close open figures and clear the workspace
close all;
clearvars;

%% Load the results and prepare the error vectors
load('results/100k-results-MatrixCompletionL1Reg.mat');

%% Open a figure
hfig = figure('Position',[100,100,1000,225]);
% hfig = figure('Position',[100,100,1000,300]);

set(hfig,'name','MatrixCompletion-MovieLens100k-l1','NumberTitle','off');

% matplotlib_colors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...
%     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'};

blue = [0, 0.4470, 0.7410];
purple = 	[0.4940, 0.1840, 0.5560];
matplotlib_colors = {purple, blue};

linewidth = 2;

%% Subfigure 1
subplot(121)

hold on;
plot(infoSHCGM.rmse_train(1:30000), 'Color', matplotlib_colors{1}, 'LineWidth',linewidth);
% plot(trainRmseWithoutBoxMean, 'Color',matplotlib_colors{2},'LineWidth',1);
plot(infoHSAGCGM.rmse_train(1:30000), 'LineWidth', linewidth, 'Color', matplotlib_colors{2});

xlabel('iteration','Interpreter','latex');
ylabel('Train RMSE','Interpreter','latex');

% ylim([0.,5]);
ax = gca;
% ax.XTick = 0:2500:10000;
% ax.YTick = -10:0.5:10;

%% Subfigure 2
subplot(122)

hl1 = plot(infoSHCGM.rmse_test(1:30000), 'Color',matplotlib_colors{1},'LineWidth',linewidth);
hold on;
hl3 = loglog(infoHSAGCGM.rmse_test(1:30000),'LineWidth',linewidth,'Color', matplotlib_colors{2});
xlabel('iteration','Interpreter','latex');
ylabel('Test RMSE','Interpreter','latex');

ax = gca;
ax.YLim = [0,6];

%% Legend
% hl = legend([hl3,hl2,hl1], 'Separable','SFW','SHCGM');
%hl = legend([hl1,hl2,hl3], 'SHCGM','SFW','Separable (us)');
hl = legend([hl1,hl3], 'SHCGM','H-SAG-CGM/v1', 'Interpreter', 'latex');
% hl.Location = 'SouthWest';
hl.Location = 'NorthEast';
hl.FontSize = 13;
hl.Interpreter = 'latex';

%% General properties
for t = 1:2
    subplot(1,2,t)
    set(gca,'TickLabelInterpreter','latex',...
        'FontSize',13,...
        'TickDir','out',...
        'LineWidth',1,'TickLength',[0.02 0.02]);
    grid on; grid minor; grid minor;
    box on
end

% sgtitle({'\textbf{matrix completion} -- $\ell_1$-regularization', '\texttt{MovieLens-100k}'}, 'Interpreter', 'latex');
% sgtitle({'$\ell_1$-regularization'}, 'Interpreter', 'latex');

sgtitle({'\textbf{matrix completion}', '\texttt{MovieLens-100k}', '$\ell_1$-regularization'}, 'Interpreter', 'latex');

%% Save Figures
% This saves pdf's perfectly cropped etc!
set(gca,'Layer','top')

% First download altmany-export_fig-d966721 and add to path
% https://github.com/altmany/export_fig
% https://github.com/altmany/export_fig/archive/refs/tags/v3.18.zip
addpath('/Users/gideon/projects/SHCGM/lib/export_fig/');

% Determine and create the path to save the figures
figPath = ['figs/',datestr(now,30),'/'];
mkdir(figPath);

figHandles = findall(groot, 'Type', 'figure');
for rr = 1:length(figHandles)
    sName = [figPath,figHandles(rr).Name];
%     savefig(figHandles(rr), sName)
%     set(0, 'CurrentFigure', figHandles(rr))
%     figHandles(rr).Color = [1,1,1];
%     if exist('export_fig','file')
%         export_fig(sName,'-pdf', '-dCompatibilityLevel=1.4');
%     end
    print(gcf, '-dpng', sName, '-r300');
end