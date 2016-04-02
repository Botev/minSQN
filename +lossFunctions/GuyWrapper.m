classdef GuyWrapper < lossFunctions.lossFunction
    properties
        net
        data
        m % n data points
        n % m number of features
        w0
    end
    methods
        
        % Constructor
        function this = GuyWrapper(net, data)
            this.net = net;
            this.data = data;
            this.m = size(data,2);
            this.w0 = net.w;
            this.n = length(net.w);
        end
        
        function f = funObj(this,w,indices)
            % Given an instantiated object called problem,
            % problem.funObj(w,indices) returns the function value at the
            % iterate w computed on the data points 'indices'.
            % If no indices are provided, the function will be computed on
            % all the data points.             
            if(nargin<3)
                indices = 1:this.m;
            end
            d = this.data(:, indices);
            f = this.net.evaluateData(w, d, d);
        end
        
        function g = gradObj(this,w,indices)
           if(nargin<3)
                indices = 1:this.m;
           end
            d = this.data(:, indices);
            [~,g] = this.net.evaluateData(w, d, d);
        end

        function Hv = hessObj(obj,w,v,indices)
           Hv = zeros(size(w));
           error('DADAD')
        end
    end
end