%% Setup 
clc
close
clear
rng(666)

% Create data
[X, Y] = generateData;

[nXDim, ~] = size(X);
[nYDim, ~] = size(Y);

% Setup hyperparameters
LR = 0.001; % Learning rate
THRESHOLD = 10e-4;
SAVE_PLOT = 0;

% Create initial w

w = rand(nYDim, nXDim + 1);

if SAVE_PLOT
    fig = figure();
end

for n = 1:30
    w = SingleLayer(X, Y, w, LR, 1, THRESHOLD);
    
    % Plot visualizaiton of result
    winv = @(x) -w(1)*x/w(2) - w(3)/w(2);
    plot(X(1, find(Y>0)), ...
        X(2, find(Y>0)), '*', ...
        X(1, find(Y<0)), ...
        X(2, find(Y<0)), '+', ...
        linspace(-2, 2),...
        arrayfun(winv, linspace(-2, 2)),'-');
    axis ([-2, 2, -2, 2], 'square');
    title(sprintf('%d epochs', n));
    if SAVE_PLOT
        if n==1
            f = getframe(fig);
            [im, map] = rgb2ind(f.cdata,256,'nodither');
        else
            f = getframe(fig);
            im(:,:,1,n) = rgb2ind(f.cdata,map,'nodither');
        end
    end
    pause(0.1)
end

 if SAVE_PLOT
    imwrite(im,map,'SingleLayer_sep.gif','DelayTime',0,'LoopCount',inf)
 end
  
 pause
 


%% Setup 
clc
close
clear
rng(666)

% Create data
[X, Y] = genNonSepData();

[nXDim, ~] = size(X);
[nYDim, ~] = size(Y);

% Setup hyperparameters
LR = 0.001; % Learning rate
THRESHOLD = 10e-4;
SAVE_PLOT = 0;


% Create initial w

w = rand(nYDim, nXDim + 1);

if SAVE_PLOT
    fig = figure();
end

for n = 1:30
    w = SingleLayer(X, Y, w, LR, 1, THRESHOLD);
    
    % Plot visualizaiton of result
    winv = @(x) -w(1)*x/w(2) - w(3)/w(2);
    plot(X(1, find(Y>0)), ...
        X(2, find(Y>0)), '*', ...
        X(1, find(Y<0)), ...
        X(2, find(Y<0)), '+', ...
        linspace(-2, 2),...
        arrayfun(winv, linspace(-2, 2)),'-');
    axis ([-2, 2, -2, 2], 'square');
    title(sprintf('%d epochs', n));
    if SAVE_PLOT
        if n==1
            f = getframe(fig);
            [im, map] = rgb2ind(f.cdata,256,'nodither');
        else
            f = getframe(fig);
            im(:,:,1,n) = rgb2ind(f.cdata,map,'nodither');
        end
    end
    pause(0.1)
end

 if SAVE_PLOT
    imwrite(im,map,'SingleLayer_non_sep.gif','DelayTime',0,'LoopCount',inf)
 end
  
 pause
 

%% Setup 
clc
close
clear
rng(666)

% Create data
[X, Y] = generateData;

[nXDim, ~] = size(X);
[nYDim, ~] = size(Y);

% Setup hyperparameters

ALPHA = 0.9;
LR = 0.01; % Learning rate
THRESHOLD = 10e-4;
EPOCHS = 80;
PRINT_INTERVAL = round(EPOCHS/20);
HIDDEN_DIM = nXDim;
SAVE_PLOT = 0;
PLOT = 1;

W = rand(HIDDEN_DIM, nXDim + 1);% [nHidden, xDim + 1]
V = rand(nYDim, HIDDEN_DIM + 1);% [yDim, nHidden + 1]

dW = zeros(HIDDEN_DIM, nXDim + 1);% [nHidden, xDim + 1]
dV = zeros(nYDim, HIDDEN_DIM + 1);% [yDim, nHidden + 1]

if PLOT
    fig = figure();
    subplot(2,1,1);
    plot(X(1, find(Y>0)), ...
        X(2, find(Y>0)), '*', ...
        X(1, find(Y<0)), ...
        X(2, find(Y<0)), '+');
    axis ([-2, 2, -2, 2], 'square');
    title('True classes');
end

for n = 1:round(EPOCHS/PRINT_INTERVAL)
    [W,V,dW,dV,O] = MulitpleLayer(X, Y, W, V, dW, dV, LR, ALPHA, PRINT_INTERVAL, HIDDEN_DIM, THRESHOLD);
    % Plot current classification.
    if PLOT       
        subplot(2,1,2);
        plot(X(1, find(O>0)), ...
            X(2, find(O>0)), '*', ...
            X(1, find(O<0)), ...
            X(2, find(O<0)), '+');
        axis ([-2, 2, -2, 2], 'square');
        title({sprintf('%d epochs', n*PRINT_INTERVAL); sprintf('error %.2f', norm(O-Y))});
        
        % Save plot if save_plot
        if SAVE_PLOT
            if n == 1
                f = getframe(fig);
                [im, map]= rgb2ind(f.cdata, 256,'nodither');
            else
                f = getframe(fig);
                im(:,:,1, n) = rgb2ind(f.cdata,map,'nodither');
            end
        end
        
        drawnow;
        pause(0.2);
    end
end

if SAVE_PLOT
    imwrite(im,map,'MultiLayer_sep.gif','DelayTime',0,'LoopCount',inf);
end

%% Setup 
clc
close
clear
rng(666)

% Create data
[X, Y] = genNonSepData;

[nXDim, ~] = size(X);
[nYDim, ~] = size(Y);

% Setup hyperparameters

ALPHA = 0.9;
LR = 0.01; % Learning rate
THRESHOLD = 10e-4;
EPOCHS = 3000;
PRINT_INTERVAL = round(EPOCHS/20);
HIDDEN_DIM = nXDim;
SAVE_PLOT = 0;
PLOT = 1;

W = rand(HIDDEN_DIM, nXDim + 1);% [nHidden, xDim + 1]
V = rand(nYDim, HIDDEN_DIM + 1);% [yDim, nHidden + 1]

dW = zeros(HIDDEN_DIM, nXDim + 1);% [nHidden, xDim + 1]
dV = zeros(nYDim, HIDDEN_DIM + 1);% [yDim, nHidden + 1]

if PLOT
    fig = figure();
    subplot(2,1,1);
    plot(X(1, find(Y>0)), ...
        X(2, find(Y>0)), '*', ...
        X(1, find(Y<0)), ...
        X(2, find(Y<0)), '+');
    axis ([-2, 2, -2, 2], 'square');
    title('True classes');
end

for n = 1:round(EPOCHS/PRINT_INTERVAL)
    [W,V,dW,dV,O] = MulitpleLayer(X, Y, W, V, dW, dV, LR, ALPHA, PRINT_INTERVAL, HIDDEN_DIM, THRESHOLD);
    % Plot current classification.
    if PLOT       
        subplot(2,1,2);
        plot(X(1, find(O>0)), ...
            X(2, find(O>0)), '*', ...
            X(1, find(O<0)), ...
            X(2, find(O<0)), '+');
        axis ([-2, 2, -2, 2], 'square');
        title({sprintf('%d epochs', n*PRINT_INTERVAL); sprintf('error %.2f', norm(O-Y))});
        
        % Save plot if save_plot
        if SAVE_PLOT
            if n == 1
                f = getframe(fig);
                [im, map]= rgb2ind(f.cdata, 256,'nodither');
            else
                f = getframe(fig);
                im(:,:,1, n) = rgb2ind(f.cdata,map,'nodither');
            end
        end
        
        drawnow;
        pause(0.2);
    end
end

if SAVE_PLOT
    imwrite(im,map,'MultiLayer_non_sep.gif','DelayTime',0,'LoopCount',inf);
end
%% Encoding
%% Setup
clc
close
clear
rng(667)

[X, Y] = generateEncodingData(8, 200);

[nXDim, ~] = size(X);
[nYDim, ~] = size(Y);

% Setup hyperparameters

ALPHA = 0.9;
LR = 0.001; % Learning rate
THRESHOLD = 10e-4;
EPOCHS = 80000;
PRINT_INTERVAL = round(EPOCHS/20);
HIDDEN_DIM = 3;
SAVE_PLOT = 0;
PLOT = 1;

W = rand(HIDDEN_DIM, nXDim + 1);% [nHidden, xDim + 1]
V = rand(nYDim, HIDDEN_DIM + 1);% [yDim, nHidden + 1]

dW = zeros(HIDDEN_DIM, nXDim + 1);% [nHidden, xDim + 1]
dV = zeros(nYDim, HIDDEN_DIM + 1);% [yDim, nHidden + 1]

if PLOT
    fig = figure();
end

for n = 1:round(EPOCHS/PRINT_INTERVAL)
    [W,V,dW,dV,O,H] = MulitpleLayer(X, Y, W, V, dW, dV, LR, ALPHA, PRINT_INTERVAL, HIDDEN_DIM, THRESHOLD);
    % Plot current classification.
    title('hej');
    if PLOT       
        for i = 1:8
          subplot(8, 1, i);
          is = find(Y(i,:)==1);
          numbers = bi2de(((sign(H)+1)/2)');
          size(find(sign(O)~=Y));
          histogram(numbers(is(1)), 0:8);
          ylabel(sprintf('%d', i));
          xlabel('H');
          if i ==1
                title({sprintf('error = %.2f', norm(O - Y)), sprintf('false = %d', size(find(sign(O)~=Y), 1)) });
          end
        end

    % Save plot if save_plot
        if SAVE_PLOT
            if n == 1
                f = getframe(fig);
                [im, map]= rgb2ind(f.cdata, 256,'nodither');
            else
                f = getframe(fig);
                im(:,:,1, n) = rgb2ind(f.cdata,map,'nodither');
            end
        end
        drawnow;
        pause(0.2);
    end
end

if SAVE_PLOT
    imwrite(im,map,'Encoded.gif','DelayTime',0,'LoopCount',inf);
end


 