function g = sigmoidGradient(z)

    g = zeros(size(z));
    g=sigmoid(z) .* (1.0-sigmoid(z));

end
