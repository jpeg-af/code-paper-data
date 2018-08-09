function [F] = cali_feature(im,w)
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
% description:   compute the calibrated feature value of the input image
%                as the criterion to determine whether it is a JPEG forgery
% 
% requires:      Matlab JPEG toolbox
% 
% INPUT
%            im: image matrix containing pixel values
%             w: weight vector for computing the variance
% 
% OUTPUT
%             F: calibrated feature value
% 
% reference:     S. Lai and R. Bohme, 
%                "Countering counter-forensics: the case of JPEG compression," 
%                IH 2011, pp. 285–298.
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% Last modified: Apr. 27th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hfs = [0  0  0  0  0  0  0  0
       0  0  0  0  0  0  0  1
       0  0  0  0  0  0  1  1
       0  0  0  0  0  1  1  1
       0  0  0  0  1  1  1  1
       0  0  0  1  1  1  1  1
       0  0  1  1  1  1  1  1
       0  1  1  1  1  1  1  1];
inds = find(hfs == true); % 28 high-frequency subbands, defined in the ref

n = 8; % DCT block size
[nH,nW] = size(im); % image size
im = double(uint8(im)); % ensure pixel values are integers with in [0,255]
im = im(1:n*floor(nH/n),1:n*floor(nW/n)); % proper cropping
I = im(1:end-8,1:end-8); I_subbands = get_subbands(I);
J = im(5:end-4,5:end-4); J_subbands = get_subbands(J);

if nargin < 2 || isempty(w), w = 1; end
F = 0;
for ii = 1:length(inds)
    F = F + abs((var(I_subbands(inds(ii),:),w) - var(J_subbands(inds(ii),:),w))./var(I_subbands(inds(ii),:),w));
end
F = F./numel(inds);

end

function [subbands] = get_subbands(im)
% returns 64 * floor(N/64) sized matrix, where N is the number of pixels 
% in the image, containing DCT coefficients of the image

subbands = im2vec(bdct(double(im)-128),8,0);

end
