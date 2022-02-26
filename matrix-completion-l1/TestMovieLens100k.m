%% Description
% This script implements the matrix completion experiment with L1
% regularization. We use MovieLens 100k dataset, make sure that you 
% % download the dataset by running DOWNLOADDATA before this test. 
% This script will save the results under the results folder. 
% Run PLOTFIG4 to generate the plots. 
% 
% [Ref] Dresdner, G. ,.......... WRITE ME .......
%
% [LY+19] Locatello, F., Yurtsever, A., Fercoq, O., Cevher, V.
% "Stochastic Conditional Gradient Method for Composite Convex Minimization"
% Advances in Neural Information Processing Systems 32 (NeurIPS 2019).
%
% Modified from the code for [LY+19] 
% contact: Gideon Dresdner

%% Fix the seed for reproducability
rng(0,'twister');

%% load data
% Atrain = dlmread('data/ml-100k/ub.base');
% Atest = dlmread('data/ml-100k/ub.test');

Atrain = dlmread('/Users/gideon/projects/SHCGM/MatrixCompletion/data/ml-100k/ub.base');
Atest = dlmread('/Users/gideon/projects/SHCGM/MatrixCompletion/data/ml-100k/ub.test');

UserID = Atrain(:,1);
MovID = Atrain(:,2);
Rating = Atrain(:,3);
UserID_test = Atest(:,1);
MovID_test = Atest(:,2);
Rating_test = Atest(:,3);
clearvars Atest Atrain

nU = max(UserID);       % # Users
nM = max(MovID);        % # Movies
nR = length(UserID);    % # Ratings

%% Clear Movies and Users that don't have any rating (Clear all zero rows and columns)
%%% 1: Clear movies and users without rating in the train dataset
A = sparse(MovID,UserID,Rating,nM,nU);
rDel = ~any(A,2);
cDel = ~any(A,1);
A( rDel, : ) = [];  %rows
A( :, cDel ) = [];  %columns
[MovID,UserID] = find(A);
clearvars A
A = sparse(MovID_test,UserID_test,Rating_test,nM,nU);
A( rDel, : ) = [];  %rows
A( :, cDel ) = [];  %columns
[MovID_test,UserID_test,Rating_test] = find(A);
nU = max(UserID);       % # Users
nM = max(MovID);        % # Movies
clearvars A rDel cDel

% Parameters
numberSample = 1000; % amount of ratings to be used at each iteration
beta1 = 7000; % problem parameter for domain diamater
beta0 = 1; % algorithm parameter for smoothing
lambda = 0.1; % regularization parameter

x0 = zeros(nM,nU); % initial point for algorithms

% Generate oracles to be used in the algorithms
[ objective, rmse_train, rmse_test, incrgradf, stogradf, gradf, proxg, lmoX, projX ] = ...
    Oracles_l1norm( numberSample,MovID,UserID,Rating,MovID_test,UserID_test,Rating_test,beta1,lambda );

% Error measures to be used
errFncs = {};
errFncs{end+1} = 'objective';
errFncs{end+1} = objective;
errFncs{end+1} = 'rmse_train';
errFncs{end+1} = rmse_train;
errFncs{end+1} = 'rmse_test';
errFncs{end+1} = rmse_test;
errFncs{end+1} = 'ell1_norm';
errFncs{end+1} = @(x) sum(abs(x), 'all');

num_iter = 1e5;

%% Run Separable SHCGM
fprintf('Separable SHCGM-Test\n');
[ ~, infoHSAGCGM ] = SeparableSHCGM( incrgradf, lmoX, proxg, beta0, x0, ...
    'maxitr', num_iter, 'errfncs', errFncs, 'printfrequency', 100 );

%% Run SHCGM
fprintf('SHCGM-Test\n');
[ ~, infoSHCGM ] = SHCGM( stogradf, lmoX, proxg, beta0, x0, ...
    'maxitr', num_iter, 'errfncs', errFncs, 'printfrequency', 100 );

%% Run Three Operator Splitting for Ground Truth
% increase # iterations if 100 is not enough
[~, infoTOS] = TOS(gradf,proxg,projX,x0,'maxitr', 100, 'errfncs', errFncs, 'printfrequency', 10);

%% Log the test setup
infoTest.beta0 = beta0;
infoTest.beta1 = beta1;
infoTest.numSample = numberSample;

%% Save results
if ~exist('results','dir'), mkdir results; end
save('results/100k-results-MatrixCompletionL1Reg.mat',...
    'infoHSAGCGM','infoSHCGM','infoTOS','infoTest',...
    '-v7.3');

%% Last edit: 18 February 2022 - Alp Yurtsever
