function [err_func, err_func_grad] = get_binary_cross_entropy()
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
err_func = @(pred, target) sum(sum(target .* softplus(-pred)  + (1 - target) .* softplus(pred)));
err_func_grad = @(pred, target) 1 ./ (1 + exp(-pred)) - target;
end

