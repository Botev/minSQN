function r = elu(x, alpha)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    r = x;
    r(x<0) = alpha*(exp(x(x<0)) - 1);
end

