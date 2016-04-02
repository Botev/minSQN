function validateGrad(varargin)
    % A functio which to check that the gradient calculations are good 
    
    this = FFNet(varargin{:});
    % Preset parameters randomly
    this.lambda = rand;
    this.dropRate = 0;
    data = randn(varargin{1},1000);
    targets = randn(varargin{end},1000);

    f = @(x) this.evaluateData(x, data, targets);
    [~,grad] = f(this.w);
    ngrad = numericalGradient(f,this.w,0.01);
    m1 = max(abs(grad-ngrad));
    ngrad = numericalGradient(f,this.w,0.001);
    m2 = max(abs(grad-ngrad));
    ngrad = numericalGradient(f,this.w,0.0001);
    m3 = max(abs(grad-ngrad));
    fprintf('All parameters gradient:\nDifference\tMAE\n0.01\t%.10f\n0.001\t%.10f\n0.0001\t%.10f\n',m1,m2,m3);
end

