function [nX, ny] = dataAugmentation(X, y, mode, lambda)
    if nargin < 3
        mode = {'scale', 'rotation', 'translation'};
    elseif nargin < 4
        lambda = 2;
    end
    
    [n, d] = size(X);
    nX = zeros(lambda*n, d);
    ny = zeros(lambda*n, 1);
    
    for i = 1:lambda:lambda*n
        nX(i,:) = X((i+lambda-1)/lambda,:); ny(i, :) = y((i+lambda-1)/lambda, :);
        img = reshape(X((i+lambda-1)/lambda, :), 16, 16);
        
        for j = 1:lambda-1
            if strcmp(mode{j},'rotation')
                tmp = rot90(img, 1);
            elseif strcmp(mode{j},'scale')
                tmp = img * 0.6;
            elseif strcmp(mode{j},'translation')
                se = translate(strel(1), [1, 1]);
                tmp = imdilate(img, se);
                tmp(tmp == -Inf) = 0;
            else
                error('The third argument must be chosen between [rotation, scale, translation]!')
            end
            nX(i+j,:) = tmp(:);
            ny(i+j, :) = y((i+lambda-1)/lambda, :);
        end
    end
end
