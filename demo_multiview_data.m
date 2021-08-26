clear; clc;
%%
addpath('TempTools');
%%
% give a dataset name for saving results
name = 'MSRC-v1';

% similar relationship based on sparse
load(sprintf('TempData/%s_Demo_multiview_SSR.mat',name));
viewNum = length(ssr_S);
wcell = cell(3*viewNum,1);
count = 1;
for ii = 1:viewNum
    wcell{count} = ssr_S{ii};
    count = count + 1;
end
clear ssr_S ssr_mea
% similar relationship based on low rank
load(sprintf('TempData/%s_Demo_multiview_LRR.mat',name));
load(sprintf('TempData/%s_Demo_multiview_LRR_Kmeans.mat',name));
for ii = 1:viewNum
    bestAcc = 0;
    for jj = 1:size(kmea.new_result,2)
        tmpMea = kmea.new_result{ii,jj};
        tmpMea = mean(tmpMea);
        if tmpMea(1) > bestAcc
            bestAcc = tmpMea(1);
            wLoc = jj;
        end
    end
    wcell{count} = results.W{ii,wLoc};
    count = count + 1;
end
clear results kmea
% similar relationship based on gaussion kernel
load(sprintf('TempData/%s_Demo_multiview_SC.mat',name));
load(sprintf('TempData/%s_Demo_multiview_SC_KNN.mat',name));
for ii = 1:viewNum
    bestAcc = 0;
    for jj = 1:size(kmea.new_result,2)
        tmpMea = kmea.new_result{ii,jj};
        tmpMea = mean(tmpMea);
        if tmpMea(1) > bestAcc
            bestAcc = tmpMea(1);
            wLoc = jj;
        end
    end
    wcell{count} = results.W{ii,wLoc};
    count = count + 1;
end
clear results

scell = cell(length(wcell),1);
for ii = 1:length(wcell)
    D = diag(sum(wcell{ii}))+eps;
    scell{ii} = diag(diag(D).^(-1/2)) * wcell{ii} * diag(diag(D).^(-1/2));
end
%%
% load feature
% the feature name should be "fea"
% fea size: nSample*mfeat
load(sprintf('%s.mat',name));
label = Y;
clear X Y 
k = length(unique(label));

param.k = 30; param.lambda = 10; param.r = 1.75; 
param.c = k; 
param.mu = 1;  param.rho = 1.1;
param.maxIter = 300; param.tol = 1e-4;
flag = 0; 
Q = 1/length(scell)^2 .* rand(length(scell),length(scell));
[G, idx_pre] = IGLmodelGPUver(Q,scell,param, flag);
new_result = ClusteringMeasure(label,idx_pre);
fprintf('[ACC, NMI, purity]=[%d,%d,%d].\n',new_result(1),new_result(2),new_result(3));
%%
rmpath('TempTools');