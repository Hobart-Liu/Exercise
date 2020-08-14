function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

list=[0.01 0.03 0.1 0.3 1 3 10 30];
%list=[0.01 0.03 0.1 0.3];
l = size(list,2);
error = zeros(l,l);
error

% I will put every error in matrix error[x,y]  list[x] is C, list[y] is sigma
for i = 1:l
	for j = 1:l
		model= svmTrain(X, y, list(i), @(x1, x2) gaussianKernel(x1, x2, list(j)));
		predictions = svmPredict(model, Xval);
		error(i,j) = mean(double(predictions ~= yval));
	end
end

% Now I try to find the min error and pick the corresponding C and sigma out. 

[val1, ind1] = min(error);
[val2, ind2] = min(val1);

error
val2


% j = ind2 and i = ind1(ind2)

ind1(ind2)
ind2

C = list(ind1(ind2))
sigma = list(ind2)


% =========================================================================

end
