function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part I
% Theta1 25 * 401
% Theta2 10 * 26

a1 = [ones(m,1) X]; % 5000*401
z2 = Theta1 * a1'; % 25 * 5000
% TMD 在这里z2转动了，a1 行为一条数据， z2 列为一条数据
a2 = sigmoid(z2); % 25 * 5000
a2 = [ones(1, m); a2]; % 26 * 5000
z3 = Theta2 * a2; % 10 * 5000
a3 = sigmoid(z3); % 10 * 5000


y_vect = zeros(num_labels, m);
for i = 1:m,
    y_vect(y(i),i) = 1;
end;


for i = 1 : m
	for k = 1 : num_labels
		J += ( (-y_vect(k, i) * log(a3(k, i)) - (1-y_vect(k, i)) * log(1-a3(k, i))) );
	end
end

J = J/m;

Theta = 0;

for j = 1 : hidden_layer_size
	for k = 1 : input_layer_size
		Theta += Theta1(j,k+1)^2;
	end
end


for j = 1 : num_labels
	for k = 1 : hidden_layer_size
		Theta += Theta2(j,k+1)^2;
	end
end


J = J + lambda / (2*m) * Theta;



% Part II

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for t = 1 : m

	delta3 = a3(:, t) - y_vect(:, t); % 10*1
	delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2(:,t));  % 25 * 1
	
	Delta2 += delta3 * a2(:,t)';
	Delta1 += delta2 * a1(t,:);

end

Theta2_grad = Delta2/m;
Theta1_grad = Delta1/m;


% Part III

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) .+ Theta2(:, 2:end) * lambda / m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) .+ Theta1(:, 2:end) * lambda / m;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
