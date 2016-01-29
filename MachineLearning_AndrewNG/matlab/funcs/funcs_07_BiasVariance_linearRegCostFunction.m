function [J, grad] = funcs_07_BiasVariance_linearRegCostFunction(X, y, theta, lambda)
    %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    %regression with multiple variables
    %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    %   cost of using theta as the parameter for linear regression to fit the 
    %   data points in X and y. Returns the cost in J and the gradient in grad

    m = length(y); % number of training examples

    J = 1/(2*m)*sum((X*theta-y).^2)+lambda/(2*m)*sum((theta(2:end).^2));

    grad = zeros(size(theta));
    grad(1)=1/m*(X*theta-y)'*X(:,1);
    for i=2:numel(theta)
        grad(i)=1/(m)*(X*theta-y)'*X(:,i)+lambda/m*theta(i);
    end
    grad = grad(:);
end
