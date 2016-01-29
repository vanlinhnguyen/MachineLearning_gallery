function [J, grad] = funcs_05_LogisticRegression_multiclass_lrCostFunction(theta, X, y, lambda)
    %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    %regularization
    %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 

    % Define sigmoid function 
    sigmoid=@(z) 1.0 ./ (1.0 + exp(-z));

    m = length(y); % number of training examples
    HTheta=sigmoid(X*theta);
    J = 1/m*sum(-y.*log(HTheta)-(1-y).*log(1-HTheta))+lambda/(2*m)*theta(2:end)'*theta(2:end);
    
    grad1=1/m*X(:,1)'*(HTheta-y); % dJ(theta)/dtheta_j,j=0
    grad2=1/m*X(:,2:end)'*(HTheta-y)+lambda/m*theta(2:end); % dJ(theta)/dtheta_j,j>0
    grad=[grad1; grad2];
end
