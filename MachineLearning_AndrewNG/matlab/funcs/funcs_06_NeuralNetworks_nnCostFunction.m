function [J,grad] = funcs_06_NeuralNetworks_nnCostFunction(nn_params, ...
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

    
    m = size(X, 1);

    % Define sigmoid function 
    sigmoid=@(z) 1.0 ./ (1.0 + exp(-z));
    sigmoidGradient = @(z) sigmoid(z).*(1-sigmoid(z)); % gradient of sigmoid function

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));


    %%    % Part 1: Feedforward the neural network and return the cost in the variable J.
 
    % Estimate actuators and outputs of each layers
    Theta_all={Theta1,Theta2,[]};
    A_all{1}=X; Z_all{1}=[];
    for layer_id=2:3
        Z_all{layer_id}=[ones(m, 1) A_all{layer_id-1}]*Theta_all{layer_id-1}';
        A_all{layer_id}=sigmoid(Z_all{layer_id});
    end
    
    % full labels of all classes (contains only 0 and 1)
    Y=zeros(size(A_all{end}));
    for label_id=1:num_labels
        y_oneclass=zeros(size(y));
        if num_labels==0
            y_oneclass(y==10)=1;
        else
            y_oneclass(y==label_id)=1;
        end
        Y(:,label_id)=y_oneclass;
    end
    
    % Estimate cost function without regularization term
    J = 1/m*sum(sum(-Y.*log(A_all{end})-(1-Y).*log(1-A_all{end})));

    % Estimate regularization terms
    reg_term=0;
    for layer_id=2:3
        reg_term = reg_term + lambda/(2*m)*sum(sum(Theta_all{layer_id-1}(:,2:end).^2));
    end  
    
    % Estimate cost function with regularization term
    J = J+reg_term;
    
    %%     % Part 2: Implement the backpropagation algorithm to compute the gradients
    %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    %         Theta2_grad, respectively. 
    
    % step 2 and 3
    delta_all{1}=[]; % there is no grad for first layer
    for layer_id=3:-1:2
        if layer_id==3
            delta_all{layer_id}=A_all{layer_id}-Y;
        else
            delta_all{layer_id}=delta_all{layer_id+1}*Theta_all{layer_id}(:,2:end).*sigmoidGradient(Z_all{layer_id});
        end
    end
    
    % step 4 and 5
    Theta_grad_all{3}=[];
    for layer_id=1:2
        A=[ones(size(A_all{layer_id},1),1) A_all{layer_id}]; % add ones to actuators
        Theta_grad_all{layer_id}=1/m*delta_all{layer_id+1}'*A;
    end

    %%     % Part 3: Implement regularization with the cost function and gradients.
    %
    %         Hint: You can implement this around the code for
    %               backpropagation. That is, you can compute the gradients for
    %               the regularization separately and then add them to Theta1_grad
    %               and Theta2_grad from Part 2.
    %
    for layer_id=1:2
        regterm_grad = [zeros(size(Theta_all{layer_id},1),1) lambda/m*Theta_all{layer_id}(:,2:end)];
        Theta_grad_all{layer_id}=Theta_grad_all{layer_id}+regterm_grad;
    end

    grad=[Theta_grad_all{1}(:); Theta_grad_all{2}(:)];
end
