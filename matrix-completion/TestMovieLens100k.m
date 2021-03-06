%% Description
% This script implements the matrix completion experiment with MovieLens
% 100k dataset. Make sure that you download the dataset by running
% DOWNLOADDATA before this test. This script will save the results under
% the results folder. Run PlotFig to generate plots.
%
% Based on code provided in [1]
%
% [1] Locatello, F., Yurtsever, A., Fercoq, O., Cevher, V. "Stochastic
% Conditional Gradient Method for Composite Convex Minimization" Advances in
% Neural Information Processing Systems 32 (NeurIPS 2019).
% https://hal.archives-ouvertes.fr/hal-02007612/
%
%
% contact: Gideon Dresdner - dgideon@inf.ethz.ch

%% Fix the seed for reproducability
rng(0,'twister');

%% load data
Atrain = dlmread('data/ml-100k/ub.base');
Atest = dlmread('data/ml-100k/ub.test');

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

numberSample = 1000; % amount of ratings to be used at each iteration

beta1 = 7000; % problem parameter for domain diamater
beta0 = 10; % algorithm parameter for smoothing

x0 = zeros(nM,nU); % initial point for algorithms

% Generate oracles to be used in the algorithms
[ rmse_train, rmse_test, gradf, proxg, lmoX, projX ] = ...
    Oracles( numberSample,MovID,UserID,Rating,MovID_test,UserID_test,Rating_test,beta1 );

% Error measures to be used
errFncs = {};
errFncs{end+1} = 'rmse_train'; 
errFncs{end+1} = rmse_train; 
errFncs{end+1} = 'rmse_test';
errFncs{end+1} = rmse_test; 
errFncs{end+1} = 'feasGap'; 
errFncs{end+1} = @(x) norm(x - proxg(x),'fro'); 

infoWithBox = {};
infoWithoutBox = {};
separableInfo = {};

num_iter = 1e4;

for TestNo = 1:10

    fprintf('Test number %d \n', TestNo);

    %% Run Separable SHCGM
    fprintf('Separable SHCGM-Test%d \n', TestNo);
    [ ~, separableInfo{end+1} ] = SeparableSHCGM( gradf, lmoX, proxg, beta0/10000, x0, ...
        'maxitr', num_iter, 'errfncs', errFncs, 'printfrequency', 100 );

    %% Run SHCGM
     fprintf('SHCGM-Test%d \n', TestNo);
     [ ~, infoWithBox{end+1} ] = SHCGM( gradf, lmoX, proxg, beta0, x0, ...
         'maxitr', num_iter, 'errfncs', errFncs, 'printfrequency', 100 );
end

save("results/100k-separable-results.mat", 'separableInfo');

%% Log the test setup
infoTest.beta0 = beta0;
infoTest.beta1 = beta1;
infoTest.numSample = numberSample;

%% Save results
% if ~exist('results','dir'), mkdir results; end
% save('results/100k-results.mat',...
%     'separableInfo','infoWithBox','infoWithoutBox','infoTest',...
%     '-v7.3');

%% Last edit: 24 October 2019 - Alp Yurtsever
