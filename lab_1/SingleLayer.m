function w = SingleLayer(X, Y, w, learning_rate, epocs, threshold, save_plot)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


 %% Run algorithm, with plots
 [~, nData] = size(X);
 X = [X; ones(1, nData)]; 
 for n = 1:epocs
    dW = Delta(w, X, Y, learning_rate);
    if norm(dW)< threshold
        break
    end
    w = w - dW;
 end



end

