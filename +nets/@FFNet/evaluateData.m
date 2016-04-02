function [err,varargout] = evaluateData(this,w,inputs, targets)
    this.w = w;
    if(nargout > 2)
        error('No more than two output argumetns are supported - error and gradient.');
    elseif(nargout > 1)
        grad = this;
    end
    M = size(this.arch,2);
    N = size(inputs,2);
    h = cell(1,M+1);
    hm = cell(1,M+1);
    mask = cell(1,M);
    h{1} = this.nodeFunc(this.W{1} * [inputs; ones(1,N)]);
    for i=2:M
        % Dropout
        if this.dropRate == 0
            % No dropout
            mask{i-1} = ones(size(h{i-1}));
        elseif this.dropRate < 1
            % Bernoulli dropout
            mask{i-1} = [(rand(size(h{i-1}) - [1 0]) > this.dropRate); ones(1,N)];
        else
            % Guassian multiplicative dropout
            mask{i-1} = [(randn(size(h{i-1}) - [1 0]) + 1); ones(1,N)];
        end
        hm{i-1} = h{i-1}.* mask{i-1};
        h{i} = this.nodeFunc(this.W{i} * [hm{i-1}; ones(1,N)]);
    end
    % Dropout
    if this.dropRate == 0
        % No dropout
        mask{M} = ones(size(h{M}));
    elseif this.dropRate < 1
        % Bernoulli dropout
        mask{M} = [(rand(size(h{M}) - [1 0]) > this.dropRate); ones(1,N)];
    else
        % Guassian multiplicative dropout
        mask{M} = [(randn(size(h{M}) - [1 0]) + 1); ones(1,N)];
    end
    hm{M} = h{M}.* mask{M};
    h{M+1} = this.W{M+1} * [hm{M}; ones(1,N)];
    err = this.errFunc(h{M+1}, targets) / N + this.lambda * this.regFunc(w,this.biasMask);
    if(nargout > 1)
        hErr = this.errFuncGrad(h{M+1}, targets) / N;
        grad.W{M+1} = hErr * [hm{M}; ones(1,N)]';
        hErr = (this.W{M+1}(:,1:end-1)' * hErr) .* mask{M} .* this.nodeFuncGrad(h{M});
        for i=M:-1:1
            if i > 1
                grad.W{i} = hErr * [hm{i-1}; ones(1,N)]';
                hErr = (this.W{i}(:,1:end-1)' * hErr) .* mask{i-1} .* this.nodeFuncGrad(h{i-1});
            else
                grad.W{i} = hErr * [inputs; ones(1,N)]';
            end
        end
        varargout{1} = grad.w + this.lambda * this.regFuncGrad(w,this.biasMask);
    end
end
