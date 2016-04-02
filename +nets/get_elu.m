function [func, func_grad] = get_elu(alpha)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    func = @(x) elu(x, alpha);
    func_grad = @(x) elu_grad(x, alpha);
end

