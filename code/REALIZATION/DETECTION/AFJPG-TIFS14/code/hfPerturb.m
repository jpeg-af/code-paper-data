function [dI] = hfPerturb(I)
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
% description:   introduce a slight perturbation to the DCT coefficients in
%                the high-frequency subbands, which has a high portion of 
%                DCT coefficients whose rounded values are integer 
%                multiples of 3
% 
% INPUT
%             I: the input image pixel value matrix
% 
% OUTPUT
%            dI: the output image
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Dec. 16th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% consider the perturbation to the coefficient across subbands
% the threshold is set to be lower than 0.6728
T = 0.5; % works fine

I = double(uint8(I));
[nH,nW] = size(I);
subbands = im2vec(bdct(double(I)-128),8,0);

dSubbands = subbands; dI = I; % initialization
for subbandId = 1:64
    subband = subbands(subbandId,:); % the current subband
    N = numel(subband);
    p3 = sum(rem(round(subband),3)==0)/N; % the percentage of coefficients which are multiples of 3
    
    if p3 > 0.6
        dSubband = subband; % initialization
        
        % in order to keep the relative proportion of coefficients -5, -4, -3, -2, -1, 1, 2, 3, 4 and 5 unchanged
        pm = [histc(round(subband),-5),histc(round(subband),-4),histc(round(subband),-3),histc(round(subband),-2),histc(round(subband),-1),...
            histc(round(subband),1),histc(round(subband),2),histc(round(subband),3),histc(round(subband),4),histc(round(subband),5)]./N;
        pm = pm./sum(pm); % normalization
        
        Nm = N*T - sum(rem(round(subband),3)==0 & abs(round(subband))>=3) - histc(round(subband),0);
        denom = (pm(3)+pm(8))/sum(pm) - 1;
        Nm = round(Nm/denom); % the number of 0 coefficients to be modified
        
        % blocks with a relatively higher tolerance of distortion will be modified first
        if sum(abs(dI-I)) == 0
            ssimInd = var(im2vec(I,8,0),1); % compute the variance instead
        else
            [~,ssimInd] = ssim_index(I,dI);
            ssimInd = sum(im2vec(PadOrCrop(ssimInd,size(I)),8,0),1);
        end
        % a simple trick to make the blocks which has 0 coefficients 
        % in this subband appear in the front during the following sorting
        ssimInd(round(subband)~=0) = -1./ssimInd(round(subband)~=0);
        [~,sortInd] = sort(ssimInd,'descend'); % sort the blocks        
        
        % prepare the data
        coefs = [];
        for k = 1:5
            Nmk = round( Nm*(pm(k)+pm(10-k+1)) ); % the number of zero coefficients to be modified to be +/-(5-k+1)
            Nm_k = round( Nm*pm(k) ); % the number of zero coefficients to be modified to be -(5-k+1)
            coefk = rand(Nmk,1)-0.5 + (5-k+1);  % sampling
            randInds = randperm(Nmk); % randomly choosing
            coefk(randInds(1:Nm_k)) = -coefk(randInds(1:Nm_k)); % flip the sign
            coefs = cat(1,coefs,coefk);
        end
        
        % modfiy the coefficients
        dSubband(sortInd(1:length(coefs))) = coefs;
        
        % update the image
        dSubbands(subbandId,:) = dSubband;
        dI = double(uint8(ibdct(vec2im(dSubbands,0,8,floor(nH/8),floor(nW/8)))+128));
    end
end

end