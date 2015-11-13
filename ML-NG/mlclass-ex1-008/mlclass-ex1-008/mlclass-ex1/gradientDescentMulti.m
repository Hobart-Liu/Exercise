function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


%Temp = [];
%Temp1 = 0;
%Temp2 = 0;
%Temp3 = 0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
%	Temp = X*theta - y;
%	Temp1 = sum(Temp.*X(:,1))*alpha/m;
%	Temp2 = sum(Temp.*X(:,2))*alpha/m;
%	Temp3 = sum(Temp.*X(:,3))*alpha/m;
	
%	theta(1) = theta(1) - Temp1;
%	theta(2) = theta(2) - Temp2;
%	theta(3) = theta(3) - Temp3;

theta = theta - alpha * (X' * (X * theta - y)) / m;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
