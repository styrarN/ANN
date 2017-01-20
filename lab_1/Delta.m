function WDelta = Delta(W_in, X_in, Y_in, learning_rate)
WDelta = learning_rate*(W_in*X_in - Y_in)*X_in.';
end
