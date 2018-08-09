function [imd] = deCalibration(im,caliT)
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
% description:   to decrease the calibration feature value
% 
% requires:      Matlab JPEG Toolbox
% 
% INPUT
%            im: image matrix containing pixel values
%         caliT: threshold, iteration stops once the calibrated feature
%                value drops below it
% 
% OUTPUT
%           imd: processed image
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Mar. 6th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% constant and parameter setting
hfs = [0  0  0  0  0  0  0  0
       0  0  0  0  0  0  0  1
       0  0  0  0  0  0  1  1
       0  0  0  0  0  1  1  1
       0  0  0  0  1  1  1  1
       0  0  0  1  1  1  1  1
       0  0  1  1  1  1  1  1
       0  1  1  1  1  1  1  1];
inds = find(hfs == true); % 28 high-frequency subbands, defined in the ref

blkSize = 8; % block size
dctm = bdctmtx(blkSize); % DCT transform matrix
[nH,nW] = size(im); % image size
im = double(uint8(im)); % ensure pixel values are integers with in [0,255]
imd = im(1:blkSize*floor(nH/blkSize),1:blkSize*floor(nW/blkSize)); % proper cropping

% some parameters for the optimization
iterMax = 100; % maximum iteration number
stepLen = 2; % step length for subgradient descent
TOL = 0.005; % tolerance

imd0 = imd; iter0 = 0;
fprintf('%s - %10.4f %10.4f %10.4f %10.4f\n',num2str(0,'%.3d'),cali_feature(imd),blk_measure(imd),blk_grad_measure(imd,1),blk_grad_measure(imd,2));
for iter = 1:iterMax
    imd1 = imd(1:end-blkSize,1:end-blkSize);
    imd2 = imd(blkSize/2+1:end-blkSize/2,blkSize/2+1:end-blkSize/2);
    
    imdX1 = im2vec(imd1,blkSize,0);
    imdX2 = im2vec(imd2,blkSize,0);
    Gr = zeros(size(imd));
    for k = 1:length(inds)
        Gr1 = subGrHFSVar(dctm(inds(k),:),imdX1);
        Gr2 = subGrHFSVar(dctm(inds(k),:),imdX2);
        
        var_diff = var(dctm(inds(k),:)*imdX1,1)-var(dctm(inds(k),:)*imdX2,1);
        
        Gr0 = zeros(size(imd));
        Gr0(1:end-blkSize,1:end-blkSize) = vec2im(Gr1,0,blkSize,size(imd1,1)/blkSize,size(imd1,2)/blkSize);
        Gr0(blkSize/2+1:end-blkSize/2,blkSize/2+1:end-blkSize/2) = Gr0(5:end-4,5:end-4)-vec2im(Gr2,0,blkSize,size(imd2,1)/blkSize,size(imd2,2)/blkSize);
        
        Gr = Gr+sign(var_diff)*Gr0;
    end
    
    Gr = Gr./max(abs(Gr(:))); % normalization
    imd = double(uint8(imd - stepLen.*Gr)); % subgradient method
    
    % record the result giving the lowest calibrated feature value
    if cali_feature(imd) < cali_feature(imd0)
        imd0 = imd;
        iter0 = iter;
    end
    
    if rem(iter,10) == 0
        fprintf('%s - %10.4f %10.4f %10.4f %10.4f\n',num2str(iter,'%.3d'),cali_feature(imd),blk_measure(imd),blk_grad_measure(imd,1),blk_grad_measure(imd,2));
    end
    
    % the iteration stops once the calibrated feature value drops below 
    % the threshold with a tolerance
    if cali_feature(imd) < caliT + TOL
        iter0 = iter;
        break;
    end
end

% if after $iterMax$ iterations, we cannot find the proper result,
% we use the one giving the lowest calibrated feautre value
if iter == iterMax
    imd = imd0;
end

fprintf('%s - %10.4f %10.4f %10.4f %10.4f\n',num2str(iter0,'%.3d'),cali_feature(imd),blk_measure(imd),blk_grad_measure(imd,1),blk_grad_measure(imd,2));

end


function [Gr] = subGrHFSVar(d,X)
% d: 1*64 sized DCT transform vector
% X: 64*L sized (L is the total number of DCT coefficients in this subband)

L = size(X,2);
y = d*X;

Gr = ones(size(X));
Gr = Gr.*((-2/L^2)*sum(y-mean(y))) + repmat((2/L)*(y-mean(y)),64,1);
Gr = Gr.*repmat(d',1,L);

end
