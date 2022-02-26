function [ xk, info ] = SeparableSHCGM( gradf, lmoX, proxg, beta0, xk, varargin)
% This implements our H-SAG-CGM method for the matrix completion task with
% inequality constraints.
%
% contact: Gideon Dresdner - dgideon@inf.ethz.ch

%% Set parameters to user specified values

% Default choices
errFncs = {};
maxitr = 1000;
printfrequency = 0;
stoptime = inf;

if (rem(length(varargin),2)==1)
    error('Options should be given in pairs');
else
    for itr=1:2:(length(varargin)-1)
        switch lower(varargin{itr})
            case 'errfncs'
                errFncs = varargin{itr+1};
            case 'maxitr'
                maxitr = varargin{itr+1};
            case 'stoptime'
                stoptime = varargin{itr+1};
            case 'printfrequency'
                printfrequency = varargin{itr+1};
            otherwise
                error(['Unrecognized option: ''' varargin{itr} '''']);
        end
    end
end

%% Preallocate data vectors
info.time = nan(maxitr,1);
for sIr = 1:2:length(errFncs)
    info.(errFncs{sIr}) = nan(maxitr,1);
end

%% Algorithm
dk = sparse(0);
clkTime = 0;
for itr = 1:maxitr
    
    % Start itration timer
    clkTimer = tic;
    
    % Main algorithm
    eta = 1/(itr+1);
    beta = beta0/sqrt(itr+1);
    
    stochastic_grad = gradf(xk);
    
    % initialize dk since we don't know the size of the gradients in
    % advance.
    if all(size(dk) == [1,1])
        dk = sparse(size(stochastic_grad,1),size(stochastic_grad,2));
    end
    
    % to overwrite the values, first set them to zero and then add in the
    % gradients which are padded with zeros wherever there is no
    % observation anyway.
    [r,c,v] = find(stochastic_grad);    
    dk([r,c]) = 0;
    dk = dk + stochastic_grad;
    
    vk = beta*dk + (xk - proxg(xk,beta)); % A = Identity;
    sXk = lmoX(vk);
    xk = xk + eta*(sXk - xk);
    
    % Stop itration timer
    clkTime = clkTime + toc(clkTimer);
    
    % save progress
    info.time(itr,1) = clkTime;
    for sIr = 1:2:length(errFncs)
        info.(errFncs{sIr})(itr,1) = errFncs{sIr+1}(xk);
    end
    
    % print progress
    if printfrequency && (mod(itr,printfrequency)==0)
        fprintf('itr = %d%4.2e', itr);
        for sIr = 1:2:min(length(errFncs),16)
            fprintf(['  \t',errFncs{sIr},' = %4.2e'],info.(errFncs{sIr})(itr,1));
        end
        fprintf('\n');
    end
    
    % check time and stop if the walltime is reached
    if clkTime >= stoptime
        info.time(itr+1:end) = [];
        for sIr = 1:2:length(errFncs)
            info.(errFncs{sIr})(itr+1:end) = [];
        end
        break;
    end
    
end

end
