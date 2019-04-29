function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

vals = [0.01,0.03,0.1,0.3,1,3,10,30];
vals_l = length(vals);

fprintf('Training %d * %d = %d models to get the best C and sigma:\n', vals_l, vals_l, (vals_l*vals_l));

error = zeros(vals_l, vals_l);

for i = 1:vals_l
	Ci = vals(i);
	for j = 1:vals_l
		sigmaj = vals(j);
		fprintf('Training for C = %d and sigma = %d:', Ci, sigmaj);
		model = svmTrain(X, y, Ci, @(x1, x2) gaussianKernel(x1, x2, sigmaj));
		error(i,j) = mean(double(svmPredict(model, Xval) ~= yval));
	end
end

fprintf('%d models have been trained and its Error matrix (C x sigma) on CV set:\n', (vals_l*vals_l));
disp(error);

[val, i] = min(min(error, [], 2));
[val, j] = min(min(error, [], 1));

fprintf('min postion %d, %d\n', i, j);

C = vals(i);
sigma = vals(j);

fprintf('Best values are: C = %d and sigma = %d\n', C, sigma);

% =========================================================================

end
