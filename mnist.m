addpath(genpath('./nets/'));
%% Set up MNIST
data = loadMNISTImages('./mnist/train-images.idx3-ubyte');
dim = 784;
net = FFNet(dim, [500, 50, 500], dim);
[net.errFunc, net.errFuncGrad] = get_binary_cross_entropy();
problem = lossFunctions.GuyWrapper(net,data);
problem.w0 = net.w;

%% Standard algorithm run
f0 = problem.funObj(problem.w0);
methods{1}.name = 'adaQN';
methods{1}.arg1 = 1e-4;
methods{1}.arg2 = 20;
methods{1}.color = 'r';
methods{2}.name = 'SGD';
methods{2}.arg1 = 1e-4;
methods{2}.arg2 = 0;
methods{2}.color = 'k';
% method_dict = {'adaQN','SGD'};%,'RES','L-RES','SDBFGS','L-SDBFGS','oBFGS','D-oBFGS','oLBFGS','D-oLBFGS'};
% step_size = [1e-6, 1e-6];
% L = [20, 0];
% method_colors = {'c','k','k--','b','b--','r','r--','g','g--','m','m--','r:'};
seed = 213;

for it=1:length(methods)
    m = methods{it};
    rng(seed);
    options = GenOptions();
    options.method = m.name;
    options.epochs = 20;
    options.batch_size = 1000;
    options.batch_size_hess = 1000;
    options.batch_size_fun = 1000;
    options.lbfgs_memory = 20;
    options.fisher_memory = 100;
    options.verbose = 2;
    options.tuning_steps = 5;
    options.damping = 1;
    options.regularization = 0;
    options.H0 = 'ADAGRAD';
    tic;
    logger = minSQN(problem,options,[m.arg1, m.arg2]);
    toc;
    f = [f0;logger.fhist];
    semilogy(0:length(f)-1,f,m.color,'LineWidth',2);
    hold on;
    drawnow();
end

%% Post-processing
xlabel('Epochs');
ylabel('Average Function Value Per Epoch');
legend(methods{1}.name, methods{2}.name,'Location','Best');
print -djpeg -r300 all_minSQN_methods.jpeg