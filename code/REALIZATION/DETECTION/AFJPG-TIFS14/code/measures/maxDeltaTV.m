function [K,deltaTVs,q] = maxDeltaTV(im)
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
% description:   obtain the maximum first-order backward finite difference
%                of total variation of the recompressed images, this value 
%                is taken as the criterion to determine whether the input 
%                image it is a JPEG forgery
% 
% INPUT
%            im: image pixel value matrix
% 
% OUTPUT
%             K: maximum first-order backward finite difference of total 
%                variation of the recompressed images
%      deltaTVs: first-order backward finite difference of total variation
%                of the recompressed images
%             q: estimated original compression quality factor
% 
% reference:     G. Valenzise, M. Tagliasacchi, and S. Tubaro, 
%                "Revealing the Traces of JPEG Compression Anti-Forensics,"
%                TIFS 2013, pp. 335-349.
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Dec. 9th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

JPGNAME = 'temp.jpg';

QFs = 45:100; % re-compression quality factors, for [50,95]
TVs = zeros(size(QFs));
for k = 1:length(QFs)
    imwrite(im,JPGNAME,'jpg','Quality',QFs(k));
    jpgI = imread(JPGNAME);
    TVs(k) = imTV(jpgI);
end
delete(JPGNAME);

% find the maximum point and estimate 
% the original compression quality factor
deltaTVs = diff(TVs);
[K, ind] = max(deltaTVs);
q = min(QFs) + ind-1;

end

function [tv] = imTV(im)
% the TV of the image (the l1 norm of the spatial first-order derivatives)

im = double(uint8(im));
tv = sum(sum(abs(im([2:end end],:) - im)));
tv = tv + sum(sum(abs(im(:,[2:end end]) - im)));

end
