function [J, grad] = funcs_04_RegularizedLogisticRegression_costFunctionReg(theta, X, y, lambda)
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 

    % Define sigmoid function
    sigmoid=@(z) 1./(1+exp(-z));
    
    m = length(y); % number of training examples

    J = 1/m*sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))+lambda/(2*m)*(theta(2:end)'*theta(2:end));
    
    grad1=1/m*X(:,1)'*(sigmoid(X*theta)-y); % dJ(theta)/dtheta_j,j=0
    grad2=1/m*X(:,2:end)'*(sigmoid(X*theta)-y)+lambda/m*theta(2:end); % dJ(theta)/dtheta_j,j>0
    grad=[grad1; grad2];
end
