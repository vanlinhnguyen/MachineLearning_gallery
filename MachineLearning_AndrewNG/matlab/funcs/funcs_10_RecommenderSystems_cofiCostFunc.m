function [J, grad] = funcs_10_RecommenderSystems_cofiCostFunc(params, Y, R, ...
    num_users, num_movies, num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

    % Unfold the U and W matrices from params
    X = reshape(params(1:num_movies*num_features), num_movies, num_features);
    Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

    XTheta=X*Theta';
    J = 1/2*sum((XTheta(R~=0)-Y(R~=0)).^2)+lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2));
    
    
    X_grad = zeros(size(X));
    Theta_grad = zeros(size(Theta));

    for i=1:size(X,1) % loop over all movies
        idx=find(R(i,:)==1);
        X_grad(i,:)=(X(i,:)*Theta(idx,:)'-Y(i,idx))*Theta(idx,:)+lambda*X(i,:);
    end

    for i=1:size(Theta,1) % loop over all users
        idx=find(R(:,i)==1);
        Theta_grad(i,:)=(X(idx,:)*Theta(i,:)'-Y(idx,i))'*X(idx,:)+lambda*Theta(i,:);
    end

    grad = [X_grad(:); Theta_grad(:)];

end