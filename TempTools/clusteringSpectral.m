function predictedLabels = clusteringSpectral(simiMatrix,k, init_num)
% clustering
n = size(simiMatrix,1);
D = diag(sum(simiMatrix,1));
L1 = D - simiMatrix;
[F] = eig1(L1,k,0);

for i = 1:init_num
    StartInd = randsrc(n,1,1:k);
    F0 = LabelFormat(StartInd);
    [Fr(:,:,i) obj(i) Q] = SpectralRotation_new(F, F0);
end;

[obj1 index] = min(obj);
Y = Fr(:,:,index);

predictedLabels = zeros(size(Y,1),1);
for ii = 1:size(Y,1)
    predictedLabels(ii) = find(Y(ii,:)==1);
end

end

% min_{Fr \in Ind, Q'Q=I}  ||Fr - F*Q ||^2
function [Fr, obj, Q] = SpectralRotation_new(F, F0)
% F:  continuous value
% F0: initial discrete value
% Fr: discrete value
% Q:  rotation matrix

[n,c] = size(F);

F(sum(abs(F),2) <= eps,:) = 1;
F = diag(diag(F*F').^(-1/2)) * F;

if nargin < 2
    StartInd = randsrc(n,1,1:c);
    G = LabelFormat(StartInd);
else
    G = F0;
end;
    
obj_old = 10^10;
for iter = 1:30
    [U, d, V] = svd(F'*G, 'econ');
    Q = U*V';
    
    M = F*Q;
    G = binarizeM(M, 'max');
    N = G - M; obj = trace(N'*N);
    if abs(obj_old - obj)/obj < 0.000001
        break;
    end;
    obj_old = obj;
end;
   
if iter == 30
    warning('Spectral Rotation does not converge');
end;

Fr = G;

end


function Y1 = LabelFormat(Y)
% Y should be n*1 or n*c

[m n] = size(Y);
if m == 1
    Y = Y';
    [m n] = size(Y);
end;

if n == 1
    class_num = length(unique(Y));
    Y1 = zeros(m,class_num);
    for i=1:class_num
        Y1(Y==i,i) = 1;
    end;
else
    [temp Y1] = max(Y,[],2);
end;

end


function B = binarizeM(M, type)
% binarize matrix M to 0 or 1

[n,c] = size(M);

B = zeros(n,c);

if strcmp(type, 'median')
    B(find(M > 0.5)) = 1;
else
    
if strcmp(type, 'min')
    [temp idx] = min(M,[],2);
elseif strcmp(type, 'max')
    [temp idx] = max(M,[],2);
end

for i = 1:n
    B(i,idx(i)) = 1;
end

end

end
