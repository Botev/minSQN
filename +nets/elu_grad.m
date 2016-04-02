function r = elu_grad(f, alpha)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    r = f + alpha;
    r(f>0) = 1;
end

