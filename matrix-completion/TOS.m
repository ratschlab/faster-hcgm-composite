function [xt, info] = TOS(gradf,proxg,proxh,x0,varargin)
%TOS Summary of this function goes here
%   Detailed explanation goes here


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

yt = full(x0);
clkTime = 0;
for itr = 1:maxitr
    % For L = 1 !!!!
    gamma = 1;
    zt = proxg(yt, gamma);
    xt = proxh(2*zt - yt - gamma.*gradf(zt));
    yt = yt - zt + xt;


    % save progress
    info.time(itr,1) = clkTime;
    for sIr = 1:2:length(errFncs)
        info.(errFncs{sIr})(itr,1) = errFncs{sIr+1}(xt);
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

