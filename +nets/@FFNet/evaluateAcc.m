function [trainAcc, testAcc] = evaluateAcc(obj,w,trainI, trainL, testI, testL)
    obj.w = w;
    M = size(obj.arch,2);
    % Train
    N = numel(trainL);
    h1 = obj.nodeFunc1(obj.W1{1} * [trainI; ones(1,N)]);
    h2 = obj.nodeFunc2(obj.W2{1} * [trainI; ones(1,N)]);
    hh = h1 .* h2 / 2;
    for i=2:M
        h1 = obj.nodeFunc1(obj.W1{i} * [hh; ones(1,N)]);
        h2 = obj.nodeFunc2(obj.W2{i} * [hh; ones(1,N)]);
        hh = h1 .* h2 / 2;
    end
    h1 = obj.W1{M+1} * [hh; ones(1,N)];
    h2 = obj.W2{M+1} * [hh; ones(1,N)];
    hh = h1 .* h2;
%     hh = obj.W1{M+1} * [hh; ones(1,N)];
    [~,predict] = max(hh);
    trainAcc = 1 - sum(predict == trainL) / N;
    % Test
    N = numel(testL);
    h1 = obj.nodeFunc1(obj.W1{1} * [testI; ones(1,N)]);
    h2 = obj.nodeFunc2(obj.W2{1} * [testI; ones(1,N)]);
    hh = h1 .* h2 / 2;
    for i=2:M
        h1 = obj.nodeFunc1(obj.W1{i} * [hh; ones(1,N)]);
        h2 = obj.nodeFunc2(obj.W2{i} * [hh; ones(1,N)]);
        hh = h1 .* h2 / 2;
    end
    h1 = obj.W1{M+1} * [hh; ones(1,N)];
    h2 = obj.W2{M+1} * [hh; ones(1,N)];
    hh = h1 .* h2;
% 	hh = obj.W1{M+1} * [hh; ones(1,N)];
    [~,predict] = max(hh);
    testAcc = 1 - sum(predict == testL) / N;
end
