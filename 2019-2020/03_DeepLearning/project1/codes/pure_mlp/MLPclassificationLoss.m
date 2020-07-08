function [f,g] = MLPclassificationLoss(w,X,y,nHidden,nLabels,dropout)

if nargin < 6
    dropout = 0.5;
end

[nInstances,nVars] = size(X);

%% Form Weights and Drops
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(...
      w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),...
      nHidden(h-1)+1, nHidden(h)); % bias
  % added Dropout
  hiddenDrop{h-1} = double(rand(nHidden(h-1)+1, nHidden(h)) < dropout) / dropout;
  offset = offset+(nHidden(h-1)+1)*nHidden(h); % bias
end
outputWeights = w(offset+1:offset+(nHidden(end)+1)*nLabels); % bias
outputWeights = reshape(outputWeights,nHidden(end)+1,nLabels); % bias


%% Init Gradients
f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

%% Compute Output
ip{1} = X * inputWeights;
fp{1} = [ones(nInstances, 1), tanh(ip{1})];
for h = 2:length(nHidden)
    ip{h} = fp{h-1} * (hiddenWeights{h-1} .* hiddenDrop{h-1});
    fp{h} = [ones(nInstances, 1), tanh(ip{h})];
end
yhat = fp{end} * outputWeights;

% Add a softmax layer
ypred = exp(yhat) ./ sum(exp(yhat), 2);

%% Back-Propagate Error
% relativeErr = yhat-y;
% f = f + sum(relativeErr.^2);
target = double(y == 1);
relativeErr = (-1) * target .* log(ypred);
f = f + sum(relativeErr); % negative log-likelihood error

if nargout > 1
    % gradient of loss w.r.t output
    %err = 2*relativeErr;
    err = ypred - target; 
    gOutput = gOutput + fp{end}'*err / nInstances; % outputWeight gradient
    
    if length(nHidden) > 1
        % Last Layer of Hidden Weights
        clear backprop
        backprop = sech(ip{end}).^2 .* (err * outputWeights(2:end,:)'); % bias
        gHidden{end} = gHidden{end} + fp{end-1}'*backprop / nInstances;
        gHidden{end} = gHidden{end} .* hiddenDrop{end}; % dropout
        
        
        % Other Hidden Layers
        for h = length(nHidden)-2:-1:1
            backprop = (backprop * hiddenWeights{h+1}(2:end,:)').*sech(ip{h+1}).^2; % bias
            gHidden{h} = gHidden{h} + fp{h}'*backprop / nInstances;
            gHidden{h} = gHidden{h} .* hiddenDrop{h}; % dropout
        end
        
        % Input Weights
        backprop = (backprop*hiddenWeights{1}(2:end,:)').*sech(ip{1}).^2; % bias
        gInput = gInput + X'*backprop / nInstances;
    else
        % Input Weights
        backprop = sech(ip{end}).^2 .* (err * outputWeights(2:end,:)') / nInstances; % bias
        gInput = gInput + X'*backprop / nInstances;
    end
end


%% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = gHidden{h-1}; %bias
        offset = offset+(nHidden(h-1)+1)*nHidden(h); %bias
    end
    g(offset+1:offset+(nHidden(end)+1)*nLabels) = gOutput(:); %bias
end
