function [dbkI] = TVsubGradJPEGDeblk(projFun,jpgI,alpha,iterN,steps,blkT,iPerturb)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------------------------------------------------------------------------
% Copyright (c) 2014 Beihang University, and GIPSA-Lab/Grenoble INP
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
% -------------------------------------------------------------------------
% If you find any bugs, please kindly report to us.
% -------------------------------------------------------------------------
% 
% description:   apply TV-based deblocking to the (post-processed) JPEG 
%                image, using the subgradient method
% 
% INPUT
%       projFun: function handler of the projection operator
%          jpgI: image pixel value matrix
%         alpha: regularization parameter
%         iterN: number of deblocking iterations
%         steps: step size for each iteration
%         dblkT: threshold for the 'blk_measure' output
%      iPerturb: whether to add a slight perturbation to some
%                high-frequency subbands
% 
% OUTPUT
%          dbkI: the deblocked image
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Jan. 6th, 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% subgradient method to solve the optimization
tvI = double(jpgI); dbkI = zeros(size(tvI));
fprintf('%s - %10.4f %10.4f %10.4f %10.4f\n',num2str(0,'%.2d'),cali_feature(tvI),blk_measure(tvI),blk_grad_measure(tvI,1),blk_grad_measure(tvI,2));
k0 = 0;
for k = 1:iterN
    % add some noise for reducing the abnormal output of the quantization
    % table estimation based detector
    if nargin >= 7 && iPerturb
        tvI = hfPerturb(tvI);
    end    
    
    % subgradient method
    [subGrTVb,~,subGrC,~] = subGr_sDerivative(tvI);
    tvI = tvI - steps(k)*(subGrTVb + alpha*subGrC);
    
    % projection
    tvI = projFun(tvI);
    
    % if there's a threshold
    if nargin >= 6 && blk_measure(tvI) < blkT
        dbkI = tvI;
        k0 = k;
        break;
    end

    % always choose the one having the smallest blk_measure value
    if k == 1 || blk_measure(tvI) < blk_measure(dbkI)
        dbkI = tvI;
        k0 = k;
    end
    
    % print out some information
    if rem(k,10) == 0
        fprintf('%s - %10.4f %10.4f %10.4f %10.4f\n',num2str(k,'%.2d'),cali_feature(tvI),blk_measure(tvI),blk_grad_measure(tvI,1),blk_grad_measure(tvI,2));
    end
end
fprintf('%s - %10.4f %10.4f %10.4f %10.4f\n',num2str(k0,'%.2d'),cali_feature(dbkI),blk_measure(dbkI),blk_grad_measure(dbkI,1),blk_grad_measure(dbkI,2));

end

function [subGrTVb,eTVb,subGrC,eC] = subGr_sDerivative(im,opt)
%
% TV_{b} = sum_{i,j}{||e_{i,j}||} = sum_{i,j}{sqrt(a_{i,j}^2+b_{i,j}^2)}
% C = |sum_{i,j}{s_{i,j}*sqrt(a_{i,j}^2+b_{i,j}^2)}|
%
% where,
%
% a_{i,j} = x_{i+1,j}+x_{i-1,n}-2*x_{i,j}
% b_{i,j} = x_{i,j+1}+x_{i,j-1}-2*x_{i,j}
% e_{i,j} = [a_{i,j},b_{i,j}]'
% s_{i,j} =  1, if pixel (i,j) is along the border of the block
%           -1, otherwise

%% s_{i,j}
[nH,nW] = size(im); S = get_S(nH,nW);

%% grad(a_{i,j},x{i,j}), grad(b_{i,j},x{i,j})

% a_{i,j}, b_{i,j}
a = (im([2:end end],:) + im([1 1:end-1],:) - 2*im);
b = (im(:,[2:end end]) + im(:,[1 1:end-1]) - 2*im);

% variation for each pixel
l2 = sqrt(a.^2+b.^2);

% the energy
eTVb = sum(l2(:)); eC = abs(sum(sum(S.*l2)));

% when ||e_{i,j}|| ~= 0
Gra = a./l2; Grb = b./l2;

% when ||e_{i,j}|| == 0, 
% its subgradient is any vector satisfying ||g|| <= 1
if nargin == 2 && opt
    % here we pick a random 2*1 sized subgradient vector for each iteration
    % g = rand(2,1); g = rand.*g./norm(g); g = g.*((rand(2,1) > .5) -.5)*2;
    % g = ones(2,1)./sqrt(2);
    % Gra(inds) = g(1); Grb(inds) = g(2);
    inds = find(l2==0);
    N = numel(inds);
    Gs = rand(3,N); norms = zeros(1,N);
    for iter = 1:N
        norms(1,iter) = norm(Gs(1:end-1,iter));
    end
    Gs = repmat(Gs(end,:),2,1).*Gs(1:end-1,:)./repmat(norms,2,1);
    Gs = Gs.*(((rand(2,N) > .5) - .5)*2); % sign
    Gs(isnan(Gs)) = 0; % just in case
    Gra(inds) = Gs(1,:)'; Grb(inds) = Gs(2,:)';
else
    % to facilitate the choosing process of the subgradient
    Gra(l2==0) = 0; Grb(l2==0) = 0;
end

%% grad(TV_{b})
subGrTVb = Gra.*(-2) + Grb.*(-2) ... % grad(e_{i,j},x_{i,j})
   + [zeros(1,size(Gra,2));Gra(1:end-1,:)] ... % grad(e_{i-1,j},x_{i,j})
   + [Gra(2:end,:);zeros(1,size(Gra,2))] ... % grad(e_{i+1,j},x_{i,j})
   + [zeros(size(Grb,1),1),Grb(:,1:end-1)] ... % grad(e_{i,j-1},x_{i,j})
   + [Grb(:,2:end),zeros(size(Grb,1),1)]; % grad(e_{i,j+1},x_{i,j})


%% grad(C)
subGrC = S.*(Gra.*(-2) + Grb.*(-2)) ... % grad(e_{i,j},x_{i,j})
        + S([1 1:end-1],:).*[zeros(1,size(Gra,2));Gra(1:end-1,:)] ... % grad(e_{i-1,j},x_{i,j})
        + S([2:end end],:).*[Gra(2:end,:);zeros(1,size(Gra,2))] ... % grad(e_{i+1,j},x_{i,j})
        + S(:,[1 1:end-1]).*[zeros(size(Grb,1),1),Grb(:,1:end-1)] ... % grad(e_{i,j-1},x_{i,j})
        + S(:,[2:end end]).*[Grb(:,2:end),zeros(size(Grb,1),1)]; % grad(e_{i,j+1},x_{i,j})
subGrC = sign(sum(sum(S.*l2))).*subGrC;

end

function [S] = get_S(nH,nW)
% S, for dividing the image pixels into two sets according to their
% positions in the DCT blocks

s = [1   1   1   1   1   1   1   1
     1   1  -1  -1  -1  -1   1   1
     1  -1  -1  -1  -1  -1  -1   1
     1  -1  -1  -1  -1  -1  -1   1
     1  -1  -1  -1  -1  -1  -1   1
     1  -1  -1  -1  -1  -1  -1   1
     1   1  -1  -1  -1  -1   1   1
     1   1   1   1   1   1   1   1];
S = repmat(s,nH/8,nW/8);

end