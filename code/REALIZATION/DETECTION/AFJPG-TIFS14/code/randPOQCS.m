function [projectedI] = randPOQCS(tvI,dctQCoefs,Q,isRelaxed)
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
% description:   for DCT histogram projection
% 
% INPUT
%           tvI: image pixel value matrix 
%     dctQCoefs: DCT coefficients of the JPEG image
%             Q: quantization table, 8 * 8 sized
%     isRelaxed: can be true or false, indicating different values for $mu$ 
%                in use for the definition of the constraint image space
%                please refer to Eq. (7) in the paper for more information
% 
% OUTPUT
%    projectedI: image after the DCT coefficient projection
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Jul. 31st, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 8; % DCT block size
[nH,nW] = size(tvI); % image size

projectedICoefs = bdct(double(tvI)-128); % transform to DCT-domain
Qmat = repmat(Q,nH/n,nW/n); % quantization table

% outliers
if isRelaxed
    outliers = round(projectedICoefs./Qmat) < quantize(dctQCoefs,Q)-1 | round(projectedICoefs./Qmat) > quantize(dctQCoefs,Q)+1;
else
    outliers = round(projectedICoefs./Qmat) ~= quantize(dctQCoefs,Q);
end
       
% rand noise to be added to DCT coefficients
randN_pos = (2*abs(rand(nH,nW)-0.5) - 0.5).*Qmat; % half closed interval [-Q/2,Q/2)
randN_zero = rand(nH,nW).*Qmat./2; % open interval (-Q/2,Q/2)
rand_neg = (-2*abs(rand(nH,nW)-0.5) + 0.5).*Qmat; % half closed interval (-Q/2,Q/2]

% randomly project the outliers to the orginal quantization bins
projectedICoefs(outliers&dctQCoefs>0) = dctQCoefs(outliers&dctQCoefs>0) + randN_pos(outliers&dctQCoefs>0);
projectedICoefs(outliers&dctQCoefs==0) = dctQCoefs(outliers&dctQCoefs==0) + randN_zero(outliers&dctQCoefs==0);
projectedICoefs(outliers&dctQCoefs<0) = dctQCoefs(outliers&dctQCoefs<0) + rand_neg(outliers&dctQCoefs<0);

% transform back to spatial-domain
projectedI = double(uint8(ibdct(projectedICoefs)+128)); % rounding & truncating

end