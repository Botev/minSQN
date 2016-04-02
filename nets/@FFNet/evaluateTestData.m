function err = evaluateTestData(this,w,inputs, targets)
    this.w = w;
    M = size(this.arch,2);
    N = size(inputs,2);
    h = this.nodeFunc(this.W{1} * [inputs; ones(1,N)]);
    for i=2:M
        h = this.nodeFunc(this.W{i} * [h; ones(1,N)]);
        if this.dropRate > 0
            h = h * this.dropRate;
        end
    end
    h = this.W{M+1} * [h; ones(1,N)];
    err = this.errFunc(h, targets) / N;
end
