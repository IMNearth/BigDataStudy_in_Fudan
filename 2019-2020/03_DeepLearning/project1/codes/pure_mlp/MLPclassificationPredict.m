function [y] = MLPclassificationPredict(w,X,nHidden,nLabels)
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(...
      w(offset+1 : offset+(nHidden(h-1)+1)*nHidden(h)),...
      nHidden(h-1)+1, nHidden(h)); % bias
  offset = offset + (nHidden(h-1)+1)*nHidden(h); % bias
end
outputWeights = w(offset+1:offset+(nHidden(end)+1)*nLabels); % bias
outputWeights = reshape(outputWeights,nHidden(end)+1,nLabels); % bias

% Compute Output
ip{1} = X*inputWeights;
fp{1} =[ones(nInstances, 1), tanh(ip{1})]; % add bias
for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = [ones(nInstances, 1), tanh(ip{h})]; % add bias
end
y = fp{end}*outputWeights;

[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
end
