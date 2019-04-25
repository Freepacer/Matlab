function [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup variables
m = size(X, 1);        
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Forwardpropagation
    X = [ones(m,1) X];
    temp1 = X * transpose(Theta1);
    temp1 = sigmoid(temp1);
    temp1 = [ones(m,1) temp1];
    temp2 = temp1 * transpose(Theta2);
    temp2 = sigmoid(temp2);
    y1=zeros(m, num_labels);
    for i=1:m
        for j=1:num_labels
            if j==y(i)
                y1(i, j)=1;
            end
        end
    end

% Cost function with regularization
    J = (1/m) * sum(sum(-y1.*log(temp2)-(1-y1).*(log(1-temp2)), 2));
    for i=1:hidden_layer_size
        for j=1:input_layer_size
            J=J + (lambda/(2*m))*(Theta1(i,j+1)^2);
        end
    end
    for i=1:num_labels
        for j=1:hidden_layer_size
            J=J + (lambda/(2*m))*(Theta2(i,j+1)^2);
        end
    end

% Backpropagation
    for t=1:m
        a_1 = transpose (X(t,:));
        z_2 = X * transpose(Theta1);
        z_2 = transpose (z_2(t,:));
        a_2 = transpose (temp1(t,:));
        z_3 = temp1 * transpose(Theta2);
        z_3 = transpose (z_3(t,:));
        a_3 = transpose (temp2(t,:));
        delta3 = a_3 - transpose(y1(t,:));
        temp3 = sigmoidGradient(z_2);
        temp4 = [1; temp3];
        delta2 = (transpose(Theta2) * delta3) .* temp4;
        delta2 = delta2 (2:end);
        Theta2_grad = Theta2_grad + delta3 * transpose(a_2);
        Theta1_grad = Theta1_grad + delta2 * transpose(a_1);
    end

% Gradient of theta matrixes with regularization
    for i=1:hidden_layer_size
        for j=1:input_layer_size+1
            if j==1
                Theta1_grad(i,j) = Theta1_grad(i,j) ./ m;
            else
                Theta1_grad(i,j) = Theta1_grad(i,j) ./ m + (lambda/m)*Theta1(i,j);
            end
        end
    end
            
    for i=1:num_labels
        for j=1:hidden_layer_size+1
            if j==1
                Theta2_grad(i,j) = Theta2_grad(i,j) ./ m;
            else
                Theta2_grad(i,j) = Theta2_grad(i,j) ./ m + (lambda/m)*Theta2(i,j);
            end
        end
    end

% Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
