function [f,g] = MLPclassificationLoss(w,X,y,nHidden,nLabels,kernel,dropout)

if nargin < 7
    dropout = 0.5;
end

[nInstances,H,W] = size(X);
convDim = H - kernel + 1;

%% Form Weights and Drops
inputWeights = reshape(w(1:kernel^2),kernel, kernel);
inputBias = w(kernel^2+1);
offset = kernel^2 + 1;
for k = 1:nHidden(1)
    fcWeight{k} = reshape(w(offset+1:offset + convDim^2), convDim, convDim);
    fcBias{k} = w(offset + convDim^2 + 1);
    offset = offset + convDim^2 + 1;
end
% ------------------------------
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
    gKernel_w = zeros(size(inputWeights));
    gKernel_b = 0;
    for k = 1:nHidden(1)
        gFullcon_w{k} = zeros(size(fcWeight{k}));
        gFullcon_b{k} = 0;
    end
    % -------------------------------
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

%% Compute Output
%convOut = zeros(nInstances, convDim, convDim);
ip{1} = zeros(nInstances, nHidden(1));
for i = 1:nInstances
    tmpOut = conv2(reshape(X(i,:,:), H, W), inputWeights, 'valid') + inputBias;
    convOut{i} = tmpOut;
    for k = 1:nHidden(1)
        ip{1}(i,k) = ...
            conv2(tanh(tmpOut), fcWeight{k}, 'valid') + fcBias{k};
    end
end
% --------------------------------
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
        
        % Full Connection Weights | To be done
        n_backprop = zeros(nInstances, nHidden(1));
        for k = 1:nHidden(1)
            n_backprop(:,k) =  backprop * hiddenWeights{1}(k+1,:)' .* sech(ip{1}(:,k)).^2;
            for i = 1:nInstances
                gFullcon_w{k} = gFullcon_w{k} + n_backprop(i, k) * tanh(convOut{i});
            end
            gFullcon_b{k} = sum(n_backprop(:,k));
        end
        
        % Convolution Weights and Bias | To be Done
        for i = 1:nInstances
            tmp = zeros(convDim, convDim);
            for k = 1:nHidden(1)
                tmp = tmp + fcWeight{k} * n_backprop(i,k);
            end
            nn_backprop{i} = tmp .* sech(convOut{i}).^2;
            gKernel_w = gKernel_w + conv2(reshape(X(i,:,:),H, W), nn_backprop{i}, 'valid');
            gKernel_b = gKernel_b + sum(nn_backprop{i}(:));
        end
        
    else
        % Full Connection Weights | To be done
        n_backprop = zeros(nInstances, nHidden(1));
        for k = 1:nHidden(1)
            n_backprop(:,k) =  err * outputWeights(k+1,:)' .* sech(ip{1}(:,k)).^2;
            for i = 1:nInstances
                gFullcon_w{k} = gFullcon_w{k} + n_backprop(i, k) * tanh(convOut{i});
            end
            gFullcon_b{k} = sum(n_backprop(:,k));
        end
        
        % Convolution Weights and Bias | To be Done
        for i = 1:nInstances
            tmp = zeros(convDim, convDim);
            for k = 1:nHidden(1)
                tmp = tmp + fcWeight{k} * n_backprop(i,k);
            end
            nn_backprop{i} = tmp .* sech(convOut{i}).^2;
            gKernel_w = gKernel_w + conv2(reshape(X(i,:,:),H, W), nn_backprop{i}, 'valid');
            gKernel_b = gKernel_b + sum(nn_backprop{i}(:));
        end
    end
end


%% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:kernel^2) = gKernel_w(:) / nInstances;
    g(kernel^2+1) = gKernel_b / nInstances;
    offset = kernel^2+1;
    for k = 1:nHidden(1)
        w(offset+1:offset + convDim^2) = gFullcon_w{k} / nInstances;
        w(offset + convDim^2+1) = gFullcon_b{k} / nInstances;
        offset = offset + convDim^2 + 1;
    end
    % ------------------------------
    for h = 2:length(nHidden)
        g(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = gHidden{h-1}; %bias
        offset = offset+(nHidden(h-1)+1)*nHidden(h); %bias
    end
    g(offset+1:offset+(nHidden(end)+1)*nLabels) = gOutput(:); %bias
end
