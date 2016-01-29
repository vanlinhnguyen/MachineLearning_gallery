function idx = funcs_09_Kmean_findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%
    K = size(centroids, 1);

    [m,n] = size(X);
    dist_all=zeros(m,K); % distant of all samples to K centroids
    for i=1:K
        temp=X-repmat(centroids(i,:),m,1);
        temp=temp.^2;
        dist_all(:,i)=sum(temp,2); 
    end

    [~,idx]=min(dist_all,[],2);

end
