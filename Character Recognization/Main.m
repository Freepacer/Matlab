clear; close all; clc

%% Setup the parameters
input_layer_size  = some_number_a;  % sqrt(some_number_a)*sqrt(some_number_a) Input Images of Digits
hidden_layer_size = some_number_b;   % some_number_b hidden units
num_labels = some_number_c;          % some_number_c labels, from 1 to 10   

% Load Training Data
load('some_name.mat'); % load data in .mat form, X and y included, X is original data matrix, y is fingerprint input by human
m = size(X, 1); % size of input data matrix

%% Initializing Pameters Randomly
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Compute Cost with Regularization (Feedforward)
lambda = 1; % regularization parameter

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

%% Training Neutral Network
options = optimset('MaxIter', 500); %set number of iterations to 500

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Using fmincg for updating cost, Theta1 and Theta2
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

%% Implement Predict
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


