function r = softplus(x,c1,c2)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
if nargin < 3
    c2 = 1;
end
if nargin < 2
    c1 = 1;
end
r = log(1 + exp(x*c1))*c2;
r(x>50/c1) = c1*c2*(x(x>50/c1));
end

