function predictions = getPredictions(this,w,inputs)
this.w = w;
M = size(this.arch,2);
N = size(inputs,2);
h = cell(1,M+1);
h{1} = this.nodeFunc(this.W{1} * [inputs; ones(1,N)]);
for i=2:M
    % Dropout
    if this.dropRate > 0 && this.dropRate < 1
        h{i-1} = h{i-1} * this.dropRate;
    end
    h{i} = this.nodeFunc(this.W{i} * [h{i-1}; ones(1,N)]);
end
% Dropout
if this.dropRate > 0 && this.dropRate < 1
    h{i-1} = h{i-1} * this.dropRate;
end
h{M+1} = this.W{M+1} * [h{M}; ones(1,N)];
predictions = this.predictFunc(h{M+1});
end

