classdef FFNet
    properties (SetAccess = immutable)
        % all arhcitetural variables (Assume all hidden layers same size)
        inputSize
        arch
        outputSize
        N           % effective number of parameters
    end
    properties
        W
        nodeFunc
        nodeFuncGrad
        errFunc
        errFuncGrad
        regFunc
        regFuncGrad
        predictFunc
        lambda
        dropRate
        maxNormConst
    end
    properties (Dependent)
        w       % vector representation of the parameters
        biasMask
    end
    methods
        % Constructor
        function obj = FFNet(inputSize,arch, outputSize)
            obj.inputSize = inputSize;
            obj.arch = arch;
            obj.outputSize = outputSize;
            obj.nodeFunc = @(x) tanh(x)*1.6;
            obj.nodeFuncGrad = @(x) 1.6  - x.^2/1.6;
            obj.errFunc = @(x,y) sum(sum((x-y).^2/2));
            obj.errFuncGrad = @(x,y) (x-y);
            obj.regFunc = @(w,b) ((w .* (~b))'*w) / 2;
            obj.regFuncGrad = @(w,b) w .* (~b);
            obj.predictFunc = @(w) w;
            obj.N = obj.arch(1) * (obj.inputSize + 1);
            for i=2:size(obj.arch,2)
                obj.N = obj.N + obj.arch(i) * (obj.arch(i-1) + 1);
            end
            obj.N = obj.N + obj.outputSize * (obj.arch(end) + 1);
            obj.lambda = 0;
            obj.dropRate = 0;
            obj.maxNormConst = 0;
            obj.w = zeros(obj.N,1);
            for i=1:size(obj.W,2)
                obj.W{i} = randn(size(obj.W{i})) / sqrt(size(obj.W{i},2));
            end
        end
        % Setter methods to keep w consistent with the Matrix parameters
        function obj = set.w(obj,val)
            if(~all([obj.N,1] == size(val)))
                error('Can not assign inconsistent size vector![Expected:(%d,%d),was (%d,%d)',[obj.N,1],size(val));
            end
            % NB - whnever there is a +1 it is for bias
            M = size(obj.arch,2);
            index = 1;
            obj.W{1} = reshape(val(index:(index - 1 + obj.arch(1) * (obj.inputSize + 1))), [obj.arch(1),  (obj.inputSize + 1)]);
            index = index + obj.arch(1) * (obj.inputSize + 1);
            for i=2:M
                obj.W{i} = reshape(val(index:(index - 1 + obj.arch(i) * (obj.arch(i-1) + 1))), [obj.arch(i),  (obj.arch(i-1) + 1)]);
                index = index + obj.arch(i) * (obj.arch(i-1) + 1);
            end
            obj.W{M+1} = reshape(val(index:(index - 1 + obj.outputSize * (obj.arch(end) + 1))), [obj.outputSize,  (obj.arch(end) + 1)]);
        end
        function w = get.w(obj)
            w = zeros(obj.N,1);
            index = 1;
            for i=1:size(obj.W,2)
                [D1,D2,D3] = size(obj.W{i});
                w(index:(index+D1*D2*D3-1),:) = reshape(obj.W{i},[D1*D2*D3,1]);
                index = index + D1*D2*D3;
            end
        end
        function bm = get.biasMask(this)
            newObj = this;
            newObj.w = zeros(size(this.w)) > 1;
            for i=1:size(this.W,2)
                newObj.W{i}(:,end) = 1>0;
            end
            bm = newObj.w;
        end
    end
	methods (Static)
        validateGrad(varargin)
    end
end