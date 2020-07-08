function [nw] = MLPclassificationFineTuning(w,X,y,nHidden,nLabels, keep_ratio)

if nargin < 6
    keep_ratio=0;
end

[nInstances,nVars] = size(X);

% Form Weights (with bias)
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars, nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(...
      w(offset+1 : offset+(nHidden(h-1)+1)*nHidden(h)),...
      nHidden(h-1)+1, nHidden(h)); % add bias
  offset = offset + (nHidden(h-1)+1)*nHidden(h); % add bias
end
outputWeights = w(offset+1 : offset+(nHidden(end)+1)*nLabels); % add bias
outputWeights = reshape(outputWeights, nHidden(end)+1, nLabels); % add bias

% Compute Output
ip{1} = X * inputWeights;
fp{1} = [ones(nInstances, 1), tanh(ip{1})]; % add bias
for h = 2:length(nHidden)
    ip{h} = fp{h-1} * hiddenWeights{h-1};
    fp{h} = [ones(nInstances, 1), tanh(ip{h})]; % activation + bias
end

newOutWeight = zeros(size(outputWeights));
newOutWeight = newOutWeight + (fp{end}'*fp{end})\fp{end}'*y;

nw = zeros(size(w)) + w;
nw(offset+1 : offset+(nHidden(end)+1)*nLabels) = ...
    keep_ratio*nw(offset+1 : offset+(nHidden(end)+1)*nLabels) ...
    + (1-keep_ratio)*newOutWeight(:);
end

