function [W, V, dW, dV, O, H] = MulitpleLayer(X, Y, W, V, dW, dV, learning_rate, alpha, epochs, nHidden, treshold)
%MULITPLELAYER Summary of this function goes here
%   Detailed explanation goes here
[xDim, nData] = size(X);
[yDim, ~] = size(Y);

sigm = @(x) 2 ./ (1 + exp(-x)) - 1;
dSigm = @(O) (1 + O) .* (1 - O) * 0.5;


for ep = 1:epochs
    % Forwardpass
    % Hidden layer
    H_star = W*[X; ones(1, nData)]; % [nHidden, nData]
    H = 2./(1 + exp(-H_star)) - 1;  % [nHidden, nData]
    
    % Output layer 
    O_star = V*[H; ones(1, nData)];% [yDim, nData]
    O = sigm(O_star);% [yDim, nData]
    
    % Backwardspass 
    O_delta = (O - Y).* dSigm(O);% [yDim, nData]
    H_delta = (V(:, 1:nHidden)' * O_delta) .* (1 + H) .*(1 - H) * 0.5;% [nHidden, nData]
   
    % Update
    dW = (dW * alpha) + (1 - alpha) * (H_delta * [X; ones(1, nData)]');% [nHidden, xDim + 1]
    dV = (dV * alpha) + (1 - alpha) * (O_delta * [H; ones(1, nData)]');% [yDim, nHidden + 1]
    W = W - dW * learning_rate;
    V = V - dV * learning_rate;
    
    if norm(dW) + norm(dV) < treshold
        break
    end
    

end



% Forwardpass
H_star = W*[X; ones(1, nData)]; % [nHidden, nData]
H = 2./(1 + exp(-H_star)) - 1;  % [nHidden, nData]

% Output layer 
O_star = V*[H; ones(1, nData)];% [yDim, nData]
O = 2./(1 + exp(-O_star)) - 1;% [yDim, nData]

end

