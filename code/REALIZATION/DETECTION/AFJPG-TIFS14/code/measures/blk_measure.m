function [K] = blk_measure(im)
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
% 
% OUTPUT
%             K: the blocking signature measure
% 
% reference:     Z. Fan and R. L. De Queiroz, "Identification of bitmap 
%                compression history: JPEG detection and quantizer 
%                estimation," IEEE Trans. Image Process., 
%                vol. 12, no. 2, pp. 230–235, 2003.
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Mar. 27th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 8; % DCT block size
[nH,nW] = size(im); % image size
im = double(uint8(im)); % ensure pixel values are integers with in [0,255]
im = im(1:n*floor(nH/n),1:n*floor(nW/n)); % proper cropping

% pixel value difference in the center of the block
As = im(4:8:end-8,4:8:end-8);
Bs = im(4:8:end-8,5:8:end-8);
Cs = im(5:8:end-8,4:8:end-8);
Ds = im(5:8:end-8,5:8:end-8);
Z1 = abs(As-Bs-Cs+Ds);

% pixel value difference in the corner of the block
Es = im(8:8:end-8,8:8:end-8);
Fs = im(8:8:end-8,9:8:end);
Gs = im(9:8:end,8:8:end-8);
Hs = im(9:8:end,9:8:end);
Z2 = abs(Es-Fs-Gs+Hs);

% construct the histograms
MAX = max(max(abs(Z1(:))),max(abs(Z2(:))));
pfreqZ1 = hist(Z1(:), -MAX:MAX);
pfreqZ1 = pfreqZ1./sum(pfreqZ1);
pfreqZ2 = hist(Z2(:), -MAX:MAX);
pfreqZ2 = pfreqZ2./sum(pfreqZ2);

% output the measure value
K = sum(abs(pfreqZ1-pfreqZ2));

end