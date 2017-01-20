function [ patterns, targets ] = generateEncodingData(nSigns, nData)
%GENERATEENCODINGDATA Summary of this function goes here
%   Detailed explanation goes here

    patterns = zeros(nSigns, nData);
    for n = 1:nData
        pattern  =  eye(8) * 2 - 1;
        target = pattern;
        
        perm = randperm(8);
        
        patterns(:,n) = pattern(perm);
        targets(:, n) = target(perm);
    end
end

