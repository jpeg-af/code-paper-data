function [KLs] = KLDivs(I1,I2)
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
% description:   measure the KL divergence of DCT histograms for 
%                64 subbands between two images in comparison
% 
% requires:      Matlab JPEG toolbox
% 
% INPUT
%            I1: image matrix of the authentic image
%            I2: image matrix of the forgery
% 
% OUTPUT
%           KLs: an 8 \times 8 matrix containing the KL divergence of DCT
%                histograms for 64 subbands between the compared two images
%  
% reference:     http://en.wikipedia.org/wiki/Kullback-Leibler_divergence
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: May. 2nd, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nH = min(size(I1,1),size(I2,1)); % image height
nW = min(size(I1,2),size(I2,2)); % image width
I1 = double(uint8(I1)); % ensure pixel values are integers with in [0,255]
I1 = I1(1:8*floor(nH/8),1:8*floor(nW/8)); % proper cropping
I2 = double(uint8(I2)); % ensure pixel values are integers with in [0,255]
I2 = I2(1:8*floor(nH/8),1:8*floor(nW/8)); % proper cropping
I1subbands = round(get_subbands(I1)); % for constructing the histogram
I2subbands = round(get_subbands(I2)); % for constructing the histogram

KLs = zeros(8,8);
for k = 1:64
    I1subband = I1subbands(k,:);
    I2subband = I2subbands(k,:);
    
    % use the same range of integers as the bin centers to construct the
    % DCT histograms
    MAX = max(max(abs(I1subband)), max(abs(I2subband)));
    I1hist = hist(I1subband,-MAX:MAX);
    I2hist = hist(I2subband,-MAX:MAX);
    
    % KL divergence
    KLs(k) = KLDiv(I1hist,I2hist);
end

end

function [subbands] = get_subbands(im)
% returns 64 * floor(N/64) sized matrix, where N is the number of pixels 
% in the image, containing DCT coefficients of the image

subbands = im2vec(bdct(double(im)-128),8,0);

end

function [dist] = KLDiv(P,Q)
% This code is written by: Nima Razavi
% For the author's information, please see: http://www.mathworks.com/matlabcentral/fileexchange/authors/31361
% We downloaded this Matlab function KLDiv.m from: http://www.mathworks.com/matlabcentral/fileexchange/20688
% 
% 
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
% have the length of one at each
% P =  n x nbins
% Q =  1 x nbins or n x nbins(one to one)
% dist = n x 1

if size(P,2)~=size(Q,2)
    error('the number of columns in P and Q should be the same');
end

if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
    error('the inputs contain non-finite values!')
end

% normalizing the P and Q
if size(Q,1)==1
    Q = Q ./sum(Q);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    temp =  P.*log(P./repmat(Q,[size(P,1) 1]));
    temp(isnan(temp))=0;% resolving the case when P(i)==0
    temp(isinf(temp))=0;
    dist = sum(temp,2);
    
    
elseif size(Q,1)==size(P,1)
    
    Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    temp =  P.*log(P./Q);
    temp(isnan(temp))=0; % resolving the case when P(i)==0
    temp(isinf(temp))=0;
    dist = sum(temp,2);
end

end
