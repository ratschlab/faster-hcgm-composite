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

%% Close open figures and clear the workspace
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

%% Open a figure
hfig = figure('Position',[100,100,850,280]);
set(hfig,'name','MatrixCompletion-MovieLens100k','NumberTitle','off');

% matplotlib_colors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...
%     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'};

blue = [0, 0.4470, 0.7410];
purple = 	[0.4940, 0.1840, 0.5560];
matplotlib_colors = {purple, blue};

%% Subfigure 1
subplot(131)

errY = [trainRmseWithBoxLow; flipud(trainRmseWithBoxUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,matplotlib_colors{1},'LineStyle','none','FaceAlpha',0.25);

errY = [trainRmseWithoutBoxLow; flipud(trainRmseWithoutBoxUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,[1,0,0],'LineStyle','none','FaceAlpha',0.15);

errY = [trainRmseSeparableLow; flipud(trainRmseSeparableUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,matplotlib_colors{2},'LineStyle','none','FaceAlpha',0.15);

hold on;
plot(trainRmseWithBoxMean, 'Color', matplotlib_colors{1}, 'LineWidth',1);
% plot(trainRmseWithoutBoxMean, 'Color',matplotlib_colors{2},'LineWidth',1);
plot(trainRmseSeparableMean, 'LineWidth', 1, 'Color', matplotlib_colors{2});

xlabel('iteration','Interpreter','latex');
ylabel('Train RMSE','Interpreter','latex');

ylim([0.,5]);
ax = gca;
ax.XTick = 0:2500:10000;
ax.YTick = -10:0.5:10;

%% Subfigure 2
subplot(132)
errY = [testRmseWithBoxLow; flipud(testRmseWithBoxUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,matplotlib_colors{1},'LineStyle','none','FaceAlpha',0.25);
hold on;
errY = [testRmseWithoutBoxLow; flipud(testRmseWithoutBoxUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,[1,0,0],'LineStyle','none','FaceAlpha',0.15);
hold on;
errY = [testRmseSeparableLow; flipud(testRmseSeparableUp)];
errX = [iterations;flipud(iterations)];
% patch(errX,errY,matplotlib_colors{2},'LineStyle','none','FaceAlpha',0.15);

plot(testRmseWithBoxMean, 'Color',matplotlib_colors{1},'LineWidth',1);
%loglog(testRmseWithoutBoxMean, 'Color',matplotlib_colors{2},'LineWidth',1);
loglog(testRmseSeparableMean,'LineWidth',1,'Color', matplotlib_colors{2});
xlabel('iteration','Interpreter','latex');
ylabel('Test RMSE','Interpreter','latex');

ylim([1,3.5]);
ax = gca;
ax.XTick = 0:2500:10000;
ax.YTick = -10:0.5:10;

%% Subfigure 3
subplot(133)
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

hl1 = plot(infeasibilityWithBoxMean,'Color',matplotlib_colors{1},'LineWidth',2);
%hl2 = plot(infeasibilityWithoutBoxMean,'Color',matplotlib_colors{2},'LineWidth',2);
hl3 = plot(infeasibilitySeparableMean, 'LineWidth', 2, 'Color', matplotlib_colors{2});

xlabel('iteration','Interpreter','latex');
ylabel('$\|X - \textrm{proj}_{[1,5]}(X) \|_F$','Interpreter','latex');

ax = gca;
ax.XTick = 10.^(-100:100);
ax.YTick = 10.^(-100:100);
ax.XScale = 'log';
ax.YScale = 'log';
ylim([1,1e4]);
xlim([1,1e4]);

%% Legend
hl = legend([hl1,hl3], 'SHCGM','H-SAG-CGM', 'Interpreter', 'latex');
hl.Location = 'SouthWest';
hl.FontSize = 13;
hl.Interpreter = 'latex';

%% General properties
for t = 1:3
    subplot(1,3,t)
    set(gca,'TickLabelInterpreter','latex',...
        'FontSize',13,...
        'TickDir','out',...
        'LineWidth',1,'TickLength',[0.02 0.02]);
    grid on; grid minor; grid minor;
    box on
end

sg = sgtitle({'\textbf{matrix completion}', '\texttt{MovieLens-100k}', 'inequality constraints'}, 'Interpreter', 'latex');
% sg.FontSize = get(groot, 'DefaultAxesFontSize') * get(groot, 'factoryAxesTitleFontSizeMultiplier'); 
sg.FontSize = 14;

%% Final values for the Table
FinalTrainRmseWithBox = trainRmseWithBox(end,:);
MeanFinalTrainRmseWithBox = mean(FinalTrainRmseWithBox);
StdFinalTrainRmseWithBox = std(FinalTrainRmseWithBox);

FinalTrainRmseWithoutBox = trainRmseWithoutBox(end,:);
MeanFinalTrainRmseWithoutBox = mean(FinalTrainRmseWithoutBox);
StdFinalTrainRmseWithoutBox = std(FinalTrainRmseWithoutBox);

FinalTestRmseWithBox = testRmseWithBox(end,:);
MeanFinalTestRmseWithBox = mean(FinalTestRmseWithBox);
StdFinalTestRmseWithBox = std(FinalTestRmseWithBox);

FinalTestRmseWithoutBox = testRmseWithoutBox(end,:);
MeanFinalTestRmseWithoutBox = mean(FinalTestRmseWithoutBox);
StdFinalTestRmseWithoutBox = std(FinalTestRmseWithoutBox);

fprintf('\nSHCGM TrainRMSE = %f +- %f \n',MeanFinalTrainRmseWithBox, StdFinalTrainRmseWithBox);
fprintf('SFW1  TrainRMSE = %f +- %f \n',MeanFinalTrainRmseWithoutBox, StdFinalTrainRmseWithoutBox);

fprintf('\nSHCGM TestRMSE  = %f +- %f \n',MeanFinalTestRmseWithBox, StdFinalTestRmseWithBox);
fprintf('SFW1  TestRMSE  = %f +- %f \n',MeanFinalTestRmseWithoutBox, StdFinalTestRmseWithoutBox);

%% Last edit: 24 October 2019 - Alp Yurtsever

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

function [amean,astd,alow,aup] = process_records(records)
    amean = mean(records,2);
    astd = std(records,0,2);
    alow = amean - astd;
    aup = amean + astd;
end
