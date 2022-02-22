%% Reset All 
close all;
clearvars;

%% Load the results and prepare the error vectors
load('results/100k-results.mat');
load('results/100k-separable-results.mat');

trainRmseWithBox = [];
testRmseWithBox = [];
infeasibilityWithBox = [];
for t = 1:length(infoWithBox)
    trainRmseWithBox(:,t) = infoWithBox{t}.rmse_train;          %#ok
    testRmseWithBox(:,t) = infoWithBox{t}.rmse_test;            %#ok
    infeasibilityWithBox(:,t) = infoWithBox{t}.feasGap;         %#ok
end
trainRmseWithoutBox = [];
testRmseWithoutBox = [];
infeasibilityWithoutBox = [];
for t = 1:length(infoWithoutBox)
    trainRmseWithoutBox(:,t) = infoWithoutBox{t}.rmse_train;    %#ok
    testRmseWithoutBox(:,t) = infoWithoutBox{t}.rmse_test;      %#ok
    infeasibilityWithoutBox(:,t) = infoWithoutBox{t}.feasGap;   %#ok
end

trainRmseSeparable = [];
testRmseSeparable = [];
infeasibilitySeparable = [];
for t = 1:length(separableInfo)
    trainRmseSeparable(:,t) = separableInfo{t}.rmse_train;    %#ok
    testRmseSeparable(:,t) = separableInfo{t}.rmse_test;      %#ok
    infeasibilitySeparable(:,t) = separableInfo{t}.feasGap;   %#ok
end


trainRmseWithBoxMean = mean(trainRmseWithBox,2);
trainRmseWithBoxStd = std(trainRmseWithBox,0,2);
trainRmseWithBoxLow = trainRmseWithBoxMean - trainRmseWithBoxStd;
trainRmseWithBoxUp = trainRmseWithBoxMean + trainRmseWithBoxStd;

trainRmseWithoutBoxMean = mean(trainRmseWithoutBox,2);
trainRmseWithoutBoxStd = std(trainRmseWithoutBox,0,2);
trainRmseWithoutBoxLow = trainRmseWithoutBoxMean - trainRmseWithoutBoxStd;
trainRmseWithoutBoxUp = trainRmseWithoutBoxMean + trainRmseWithoutBoxStd;

testRmseWithBoxMean = mean(testRmseWithBox,2);
testRmseWithBoxStd = std(testRmseWithBox,0,2);
testRmseWithBoxLow = testRmseWithBoxMean - testRmseWithBoxStd;
testRmseWithBoxUp = testRmseWithBoxMean + testRmseWithBoxStd;
 
testRmseWithoutBoxMean = mean(testRmseWithoutBox,2);
testRmseWithoutBoxStd = std(testRmseWithoutBox,0,2);
testRmseWithoutBoxLow = testRmseWithoutBoxMean - testRmseWithoutBoxStd;
testRmseWithoutBoxUp = testRmseWithoutBoxMean + testRmseWithoutBoxStd;

infeasibilityWithBoxMean = mean(infeasibilityWithBox,2);
infeasibilityWithBoxStd = std(infeasibilityWithBox,0,2);
infeasibilityWithBoxLow = infeasibilityWithBoxMean - infeasibilityWithBoxStd;
infeasibilityWithBoxUp = infeasibilityWithBoxMean + infeasibilityWithBoxStd;

infeasibilityWithoutBoxMean = mean(infeasibilityWithoutBox,2);
infeasibilityWithoutBoxStd = std(infeasibilityWithoutBox,0,2);
infeasibilityWithoutBoxLow = infeasibilityWithoutBoxMean - infeasibilityWithoutBoxStd;
infeasibilityWithoutBoxUp = infeasibilityWithoutBoxMean + infeasibilityWithoutBoxStd;

[trainRmseSeparableMean, trainRmseSeparableStd, trainRmseSeparableLow, trainRmseSeparableUp] = process_records(trainRmseSeparable);
[testRmseSeparableMean, testRmseSeparableStd, testRmseSeparableLow, testRmseSeparableUp] = process_records(testRmseSeparable);
[infeasibilitySeparableMean, infeasibilitySeparableStd, infeasibilitySeparableLow, infeasibilitySeparableUp] = process_records(infeasibilitySeparable);

iterations = (1:length(trainRmseWithBoxMean))';

%% Load l1 results
load('/Users/gideon/projects/faster-scgm-composite/matrix-completion/results/100k-results-MatrixCompletionL1Reg.mat');


%% New Figure / Global Properties
hfig = figure('Position',[100,100,1000,700]);
set(hfig,'name','MatrixCompletion-MovieLens100k','NumberTitle','off');

% matplotlib_colors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...
%     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'};
blue = [0, 0.4470, 0.7410];
purple = 	[0.4940, 0.1840, 0.5560];
matplotlib_colors = {purple, blue};

linewidth = 2;

%% Subfigure 1
xx = 0.1;
yy = 0.05;

subplot_tight(2,3,1, [xx, yy]);

errY = [trainRmseWithBoxLow; flipud(trainRmseWithBoxUp)];
errX = [iterations;flipud(iterations)];

errY = [trainRmseWithoutBoxLow; flipud(trainRmseWithoutBoxUp)];
errX = [iterations;flipud(iterations)];

errY = [trainRmseSeparableLow; flipud(trainRmseSeparableUp)];
errX = [iterations;flipud(iterations)];

hold on;
plot(trainRmseWithBoxMean, 'Color', matplotlib_colors{1}, 'LineWidth',linewidth);
plot(trainRmseSeparableMean, 'LineWidth', linewidth, 'Color', matplotlib_colors{2});

xlabel('iteration','Interpreter','latex');
ylabel('Train RMSE','Interpreter','latex');

ylim([0.,5]);
ax = gca;
ax.XTick = 0:2500:10000;
ax.YTick = -10:0.5:10;

    set(gca,'TickLabelInterpreter','latex',...
        'FontSize',13,...
        'TickDir','out',...
        'LineWidth',1,'TickLength',[0.02 0.02]);
    grid on; grid minor; grid minor;
    box on

%% Subfigure 2

subplot_tight(2,3,2, [xx, yy]);
errY = [testRmseWithBoxLow; flipud(testRmseWithBoxUp)];
errX = [iterations;flipud(iterations)];
hold on;
errY = [testRmseWithoutBoxLow; flipud(testRmseWithoutBoxUp)];
errX = [iterations;flipud(iterations)];
hold on;
errY = [testRmseSeparableLow; flipud(testRmseSeparableUp)];
errX = [iterations;flipud(iterations)];

plot(testRmseWithBoxMean, 'Color',matplotlib_colors{1},'LineWidth',linewidth);
loglog(testRmseSeparableMean,'LineWidth',linewidth,'Color', matplotlib_colors{2});
xlabel('iteration','Interpreter','latex');
ylabel('Test RMSE','Interpreter','latex');

ylim([1,3.5]);
ax = gca;
ax.XTick = 0:2500:10000;
ax.YTick = -10:0.5:10;

    set(gca,'TickLabelInterpreter','latex',...
        'FontSize',13,...
        'TickDir','out',...
        'LineWidth',1,'TickLength',[0.02 0.02]);
    grid on; grid minor; grid minor;
    box on

%% Subfigure 3
subplot_tight(2,3,3, [xx, yy]);
hold on;

errY = [infeasibilityWithBoxLow; flipud(infeasibilityWithBoxUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,matplotlib_colors{1},'LineStyle','none','FaceAlpha',0.25);

errY = [infeasibilityWithoutBoxLow; flipud(infeasibilityWithoutBoxUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,[1,0,0],'LineStyle','none','FaceAlpha',0.15);

errY = [infeasibilitySeparableLow; flipud(infeasibilitySeparableUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,matplotlib_colors{2},'LineStyle','none','FaceAlpha',0.15);

hl1 = plot(infeasibilityWithBoxMean,'Color',matplotlib_colors{1},'LineWidth',linewidth);
%hl2 = plot(infeasibilityWithoutBoxMean,'Color',matplotlib_colors{2},'LineWidth',2);
hl3 = plot(infeasibilitySeparableMean, 'LineWidth', linewidth, 'Color', matplotlib_colors{2});

    set(gca,'TickLabelInterpreter','latex',...
        'FontSize',13,...
        'TickDir','out',...
        'LineWidth',1,'TickLength',[0.02 0.02]);
    grid on; grid minor; grid minor;
    box on

xlabel('iteration','Interpreter','latex');
ylabel('$\|X - \textrm{proj}_{[1,5]}(X) \|_F$','Interpreter','latex');

ax = gca;
ax.XTick = 10.^(-100:100);
ax.YTick = 10.^(-100:100);
ax.XScale = 'log';
ax.YScale = 'log';
ylim([1,1e4]);
xlim([1,1e4]);

%% Fixup First Row
% for t = 1:3
%     sp(t) = subplot_tight(2,3,t);
%     set(gca,'TickLabelInterpreter','latex',...
%         'FontSize',13,...
%         'TickDir','out',...
%         'LineWidth',1,'TickLength',[0.02 0.02]);
%     grid on; grid minor; grid minor;
%     box on
% end

hl = legend([hl1,hl3], 'SHCGM','H-SAG-CGM/v1', 'Interpreter', 'latex');
hl.Location = 'SouthWest';
hl.FontSize = 13;
hl.Interpreter = 'latex';

%% Second Row / Subfigure 4 / l1 norm

xx_prime = xx + 0.075;
yy_prime = yy;

subplot_tight(2,2,3, [xx_prime, yy_prime]);
hold on;
plot(infoSHCGM.rmse_train(1:30000), 'Color', matplotlib_colors{1}, 'LineWidth',linewidth);
% plot(trainRmseWithoutBoxMean, 'Color',matplotlib_colors{2},'LineWidth',1);
plot(infoHSAGCGM.rmse_train(1:30000), 'LineWidth', linewidth, 'Color', matplotlib_colors{2});

xlabel('iteration','Interpreter','latex');
ylabel('Train RMSE','Interpreter','latex');

ax = gca;

    set(gca,'TickLabelInterpreter','latex',...
        'FontSize',13,...
        'TickDir','out',...
        'LineWidth',1,'TickLength',[0.02 0.02]);
    grid on; grid minor; grid minor;
    box on

%% Second Row / Subfigure 4 / l1 norm
subplot_tight(2,2,4, [xx_prime, yy_prime]);

hl1 = plot(infoSHCGM.rmse_test(1:30000), 'Color',matplotlib_colors{1},'LineWidth',linewidth);
hold on;
hl3 = loglog(infoHSAGCGM.rmse_test(1:30000),'LineWidth',linewidth,'Color', matplotlib_colors{2});
xlabel('iteration','Interpreter','latex');
ylabel('Test RMSE','Interpreter','latex');

ax = gca;
ax.YLim = [0,6];

    set(gca,'TickLabelInterpreter','latex',...
        'FontSize',13,...
        'TickDir','out',...
        'LineWidth',1,'TickLength',[0.02 0.02]);
    grid on; grid minor; grid minor;
    box on

%% Fixup Second Row
% for t = 3:4
%     sp(t+1) = subplot_tight(2,2,t);
%     set(gca,'TickLabelInterpreter','latex',...
%         'FontSize',13,...
%         'TickDir','out',...
%         'LineWidth',1,'TickLength',[0.02 0.02]);
%     grid on; grid minor; grid minor;
%     box on
% end

%% Legend
hl = legend([hl1,hl3], 'SHCGM','H-SAG-CGM/v1', 'Interpreter', 'latex');
hl.Location = 'NorthEast';
hl.FontSize = 13;
hl.Interpreter = 'latex';

%% Titles
sg = sgtitle({'\textbf{matrix completion}', '\texttt{MovieLens-100k}', 'inequality constraints'}, 'Interpreter', 'latex');
% sp(4).Position(1:2)
annotation('textbox','Position',[0.3500 0.1900 0.3 0.3],'HorizontalAlignment','center','LineStyle','none', 'String', {'\textbf{matrix completion}', '\texttt{MovieLens-100k}', '$\ell_1$-regularization'}, 'Interpreter', 'latex', 'fontsize', 15)

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

%% Helper Functions
function [amean,astd,alow,aup] = process_records(records)
    amean = mean(records,2);
    astd = std(records,0,2);
    alow = amean - astd;
    aup = amean + astd;
end