function [nw] = MLPclassificationFineTuning(w,X,y,nHidden,nLabels, kernel, keep_ratio)

if nargin < 7
    keep_ratio=0;
end

[nInstances,H,W] = size(X);
convDim = H - kernel + 1;

% Form Weights and Drops
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
  offset = offset+(nHidden(h-1)+1)*nHidden(h); % bias
end
outputWeights = w(offset+1:offset+(nHidden(end)+1)*nLabels); % bias
outputWeights = reshape(outputWeights,nHidden(end)+1,nLabels); % bias


% Compute Output
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
    ip{h} = fp{h-1} * hiddenWeights{h-1};
    fp{h} = [ones(nInstances, 1), tanh(ip{h})];
end

newOutWeight = zeros(size(outputWeights));
newOutWeight = newOutWeight + (fp{end}'*fp{end})\fp{end}'*y;

nw = zeros(size(w)) + w;
nw(offset+1 : offset+(nHidden(end)+1)*nLabels) = ...
    keep_ratio*nw(offset+1 : offset+(nHidden(end)+1)*nLabels) ...
    + (1-keep_ratio)*newOutWeight(:);
end

