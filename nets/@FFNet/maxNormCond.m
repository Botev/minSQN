function w = maxNormCond(this,w)
    if this.maxNormConst ~= 0
        this.w = w;
        for i=1:size(this.W,2)
            norms = sum(this.W{i}(:,1:end-1).^2,2);
            index = norms > (this.maxNormConst^2);
            if sum(index) > 0
                this.W{i}(index,1:end-1) = bsxfun(@rdivide, this.W{i}(index,1:end-1), sqrt(norms(index)) / this.maxNormConst);
            end
        end
        w = this.w;
    end
end

