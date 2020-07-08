clear; clc; rng(2020);

%% Prepare data
load digits.mat % load X,Xtest,Xvalid  Y,Ytest,Yvalid

%[X, y] = dataAugmentation(X, y, {'translation'}, 2);

[n,d] = size(X);
nLabels = max(y);

% From 1D to 2D matrix with shape(n, nLabels)
% yExpanded[true_label] = 1, all others are -1
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

%% Choose network structure
nHidden = [512, 256];

%% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+(nHidden(h-1)+1)*nHidden(h); % bias
end
nParams = nParams+(nHidden(end)+1)*nLabels; % bias
w = randn(nParams,1);

% No regulation on bias vector
bias_mask = ones(size(w));
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    bias_mask(nParams+1:nParams + nHidden(h)) = 0;
    nParams = nParams + (nHidden(h-1)+1)*nHidden(h); % add bias
end
bias_mask(nParams+1:nParams+nLabels) = 0;
nParams = nParams + (nHidden(end)+1)*nLabels; % add bias

%% Train with stochastic gradient -> momentum gradient descent
num_epochs = 20; maxIter = num_epochs*n;
batch_size = 20; num_batches = floor(n/batch_size);

train_mode = "l2reg-momentum";
stepSize=2e-3; beta=0.9; lambda=0.08; decay_factor=0.96;

fprintf(" ----------- %s ----------- \n", train_mode);
fprintf("total train_examples:[%d]\n", n);
fprintf("num_epoch:[%d]\tbatch_size:[%d]\n", num_epochs, batch_size);
fprintf("nHidden: "); disp(nHidden);
fprintf("stepSize:[%f]\tdecay:[%.3f]\n", stepSize, decay_factor);
fprintf("beta:[%.4f]\t\tlambda:[%.3f]\n", beta, lambda)
fprintf(" ---------------------------------------- \n")

funObj = @(w,idx)MLPclassificationLoss(...
    w,X(idx,:),yExpanded(idx,:),nHidden,nLabels); % function_handle

pw = zeros(size(w));
tic
for ep = 1:num_epochs
    if ep == 1 || mod(ep,round(num_epochs/10)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        errorRate = sum(yhat~=yvalid)/t;
        fprintf('Training epoch = %d, validation error = %f\n',ep,errorRate);
    end
    
    perm = randperm(n);
    for iter = 1:num_batches
        idx = perm((iter-1)*batch_size+1 : iter*batch_size);
        
        [~,g] = funObj(w,idx);
        
        tmp = w;
        if train_mode == "momentum"
            w = w - stepSize * g + beta*(w - pw); % momentum
        elseif train_mode == "l2reg-momentum"
            w = w - stepSize * (g + lambda*w.*bias_mask) + beta*(w - pw);
        elseif train_mode == "sgd"
            w = w - stepSize * g; % pure sgd
        end
        pw = tmp;
    end
    
    stepSize = stepSize * decay_factor;
end
toc

%% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);

%% Fine-Tuning (for the last layer) Added
nw = MLPclassificationFineTuning(w, X, yExpanded, nHidden, nLabels);
yhat = MLPclassificationPredict(nw, Xtest,nHidden,nLabels);
fprintf('----------------------------------------------\n')
fprintf('After -- Test error with final model = %f\n',sum(yhat~=ytest)/t2);
