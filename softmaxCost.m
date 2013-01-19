function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector   N = 784
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set  M = 60000     784 x 60000
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta  numClasses = 10, inputsize = 784
theta = reshape(theta, numClasses, inputSize);  %row corrospond to a row of theta

numCases = size(data, 2);  %60000

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


a = exp(theta * data);

b = sum(a);
b = vertcat(b,b,b,b,b);
p = a ./ b;

%{
c = zeros(inputSize,numCases);
for i = 1: numCases
    c(:,i) = a(:,i) ./ b(:,i);
end

p = zeros(numClasses,numCases);   %  10 x 60000
for i = 1 : numCases
    for j = 1 : numClasses
        temp1 = exp(theta(j,:) * data(:,i));  %2 corrospond to jth, 1 corrospond to ith
        temp2 = sum(exp(theta * data(:,i)));
        p(j,i) = temp1./temp2;
    end
end
yyy =  p - c;
%}

temp3 = groundTruth - p;
thetagrad = (-data * temp3')./numCases + lambda.*theta';
thetagrad = thetagrad';
temp4 = zeros(numCases,1);
logp = log(p);
for i = 1 : numCases
    temp4(i) = groundTruth(:,i)' * logp(:,i);
end

temp5 = sum(sum(theta.^2));
cost = -sum(temp4)./numCases + (lambda/2).* temp5;

%thetagrad 
%fprintf('debug  %10d\n',sum(p));








% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end


