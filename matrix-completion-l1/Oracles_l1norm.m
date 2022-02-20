function [ objective, rmse_train, rmse_test, incrGradf, stochGradf, Gradf, proxg, lmoX, projX ] = Oracles_l1norm( numberSample,MovID,UserID,Rating,MovID_test,UserID_test,Rating_test,beta1,lambda )
%% RMSE
nU = max(UserID);       % # Users
nM = max(MovID);        % # Movies
nR = length(UserID);    % # Ratings

ind_train = sub2ind([nM,nU],MovID, UserID);
ind_test = sub2ind([nM,nU],MovID_test, UserID_test);

rmse_train = @(x) sqrt(mean((x(ind_train) - Rating).^2));
rmse_test = @(x) sqrt(mean((x(ind_test) - Rating_test).^2));
objective = @(x) sum((x(ind_train) - Rating).^2) + lambda*norm(x(:),1);

%% gradient

incrGradf = @(x) evalGradientIncr(x);

    function xout = evalGradientIncr(x)
        sample = sort(randsample(nR,numberSample));
        ind_sample = ind_train(sample);
        gf_forw = (x(ind_sample)-Rating(sample)) .* 2;
        [I,J] = ind2sub([nM,nU],ind_sample);
        xout = sparse(I,J,gf_forw,nM,nU);
    end

stochGradf = @(x) incrGradf(x) .* (nR/numberSample);

Gradf = @(x) evalGradient(x);
    function xout = evalGradient(x)
        gf_forw = (x(ind_train) - Rating) .* 2;
        xout = sparse(MovID,UserID,gf_forw,nM,nU);
    end


%% proximal operator
% proxg = @(x,gamma) min(max(x,1),5);

proxg = @(x, gamma) sign(x) .* max(abs(x) - lambda*gamma, 0);


%% linear minimization oracle
lmoX = @(x) evalLmo(x);

    function Xout = evalLmo(A)
        % This dunction implements the power method for solving LMO
        tolerance = 1e-4; % decrease this if needed!
        maxiter = 1000;    % increase this if needed!
        x = randn(size(A,2),1);
        x = x/norm(x);
        e = 1;
        for dummy = 1:maxiter
            e_old = e;
            Ax = A*x;
            x = A'*Ax;
            normx = norm(x);
            e = normx/norm(Ax);
            x = x/normx;
            if abs(e - e_old) <= tolerance*e
                break;
            elseif dummy == maxiter
                warning('LMO did not converge. Increase the maximum number of iterations of LMO subsolver.');
            end
        end
        lvec = Ax/norm(Ax);
        rvec = x;
        Xout = lvec*(-beta1)*rvec';
    end

%% Below is the less efficient but very reliable LMO implementation with SVDS of Matlab
%     function xout = evalLmo(x)
%         optsSvds.Tolerance = 1e-6;
%         optsSvds.SubspaceDimension = 5;
%         for dummy = 1:10
%             [lvec, ~, rvec,FLAG] = svds(x, 1, 'L', optsSvds);
%             optsSvds.SubspaceDimension = 2*optsSvds.SubspaceDimension;
%             if ~FLAG
%                 break;
%             end
%         end
%         if (dummy == 10) && FLAG
%             warning('LMO Solver accuracy problem.');
%         end
%         xout = lvec*(-beta1)*rvec';
%     end

%% projection
projX = @(x,gamma) evalProj(x); % gamma is dummy - not used
    function xout = evalProj(x)
        [U,S,V] = svd(x,0);
        S = spdiags(projL1norm(diag(S),beta1),0,size(U,2),size(V,1));
        xout = U*S*V';
    end

end

function xout = projL1norm(x, kappa)

sx   = sort(abs(nonzeros(x)), 'descend');
csx  = cumsum(sx);
nidx = find( csx - (1:numel(sx))'.*[sx(2:end); 0] >= kappa + 2*eps(kappa),1);
if ~isempty(nidx)
    dx   = ( csx(nidx) - kappa ) /nidx;
    xout = x.*( 1 - dx./ max(abs(x), dx) );
else
    xout = x;
end

end
