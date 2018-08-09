function [K] = blk_grad_measure(im,lambda)
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
% description:   measure the presence of blocking artifacts
%                by JPEG compression
% 
% INPUT
%            im: image matrix containing pixel values
%        lambda: norm parameter
% 
% OUTPUT
%             K: the measure of gradient aware blockiness
%  
% reference:     W. Fan, K. Wang, F. Cayre, and Z. Xiong,
%                "A variational approach to JPEG anti-forensics",
%                ICASSP, 2013.
%                C. Ullerich and A. Westfeld, "Weakness of MB2", IWDW
%                vol. 5041, pp. 127-142, 2007.
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Mar. 27th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 8; % DCT block size
[nH,nW] = size(im); % image size
im = double(uint8(im)); % ensure pixel values are integers with in [0,255]
im = im(1:n*floor(nH/n),1:n*floor(nW/n)); % proper cropping
I = im; J = I(5:end,5:end); % calibration

% output the measure value
K = abs(blk_grad(I,lambda) - blk_grad(J,lambda));

end

function [bgr] = blk_grad(im,lambda)

n = 8; % DCT block size
[nH,nW] = size(im); % image size

%% vertical
r = floor((nH-2)/8);
g_ni_1 = im(n-1:n:n*r-1,:);
g_ni = im(n:n:n*r,:);
g_ni_01 = im(n+1:n:n*r+1,:);
g_ni_02 = im(n+2:n:n*r+2,:);
bgr = sum(sum((abs(g_ni_1-3*g_ni+3*g_ni_01-g_ni_02)).^lambda));

%% horizontal
c = floor((nW-2)/n);
g_nj_1 = im(:,n-1:n:n*c-1);
g_nj = im(:,n:n:n*c);
g_nj_01 = im(:,n+1:n:n*c+1);
g_nj_02 = im(:,n+2:n:n*c+2);
bgr = bgr + sum(sum((abs(g_nj_1-3*g_nj+3*g_nj_01-g_nj_02)).^lambda));

%% gradient aware blockiness
bgr = bgr/(nW*r+nH*c);

end