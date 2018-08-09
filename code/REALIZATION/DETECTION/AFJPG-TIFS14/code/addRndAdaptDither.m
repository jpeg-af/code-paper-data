function [ditheredCompTVSubband] = addRndAdaptDither(tvSubband,Q,isDC)
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
% function:      DCT histogram smoothing with a local Laplacian model
% 
%                Here DCT histograms are constructed using the same 
%                range of integers as the bin centers.
% 
% INPUT
%     tvSubband: input DCT coefficient for a certain subband here
%                the subband is supposed to be from a postprocessed JPEG 
%                image using the first-round TV-based deblocking method
%             Q: quantization step
%          isDC: whether this subband is from the DC component
% 
% OUTPUT
% ditheredCompTVSubband: processed DCT coefficients with a smoothed DCT
%                histogram here noise is added randomly without any 
%                consideration of the image information in the spatial domain
%
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% Last modified: Aug. 20th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

borderTMAX = 3; % a threshold for distinguishing quantization bins 0 which don't need search for a limit for data fitting

compTVSubband = round(tvSubband./Q)*Q; % quantize DCT coefficients
MAX = max(abs(tvSubband));
MAX = round(MAX/Q)*Q + (round(MAX/Q)<0)*floor(Q/2) + (round(MAX/Q)==0)*(ceil(Q/2)-1) + (round(MAX/Q)>0)*(ceil(Q/2)-1);
coefRange = -MAX:MAX;
tvHist = hist(round(tvSubband),coefRange); % integers are used as the bin centers while constructing the DCT histograms
qBinCenters = unique(round(coefRange./Q)); % the center of the quantization bins
if qBinCenters == 0
    compTVqBinHist = numel(compTVSubband);
else
    compTVqBinHist = hist(round(compTVSubband./Q),qBinCenters); % coefficient populations in different quantization bins
end
ditheredCompTVSubband = compTVSubband; % initialization

%% quantization bin 0
qBin = 0;
if isDC
    isUni = true;
else
    isUni = false;    
    if numel(qBinCenters) <= 1
        lambdaT = 1;
    else
        borderT = max(compTVqBinHist(ceil(end/2)-1)./Q, compTVqBinHist(ceil(end/2)+1)./Q); % say uniform distribution in the neighboring bins
        if borderT < borderTMAX
            lambdaT = 1;
        else
            lambdas = [1e-3,0.005:0.005:1]; % search the threshold for the parameter lambda
            histZeroMat = compTVqBinHist(ceil(end/2)).*get_pmf(Q,0,lambdas);
            if rem(Q,2) == 1 % Q is an odd number
                lambdaT = max(lambdas(histZeroMat(:,1) >= borderT));
            else % Q is an even number
                lambdaT = max(lambdas(histZeroMat(:,2) >= histZeroMat(:,1)+borderT/2));
            end
            if isempty(lambdaT), lambdaT = 1; end
        end
    end
    
    % fitting, in order to find the lambda
    weights = 1./(abs(round(coefRange./Q) - qBin) + 1).^1;
    func = @(lambda,xdata) pmfLaplacian(lambda,xdata,sqrt(weights));
    if lambdaT == 1e-3
        lambdaZero = lambdaT;
    else
        lambdaZero = lsqcurvefit(func,(lambdaT+1e-3)/2,coefRange,tvHist.*sqrt(weights)./sum(tvHist),1e-3,lambdaT,optimset('Display','off'));
    end
    pmfZero = get_pmf(Q,0,lambdaZero); % pmf
end

% add the dithering signal
% (randomly, without considering the image spatial-domain information)
yindex = find(round(compTVSubband./Q) == qBin);
if isUni
    noises = get_dither(length(yindex),Q,0); % uniform distribution
else
    noises = get_dither(length(yindex),Q,0,lambdaZero); % Laplacian distribution
end
ditheredCompTVSubband(yindex) = compTVSubband(yindex) + noises; % add the noise

%% quantization bins on the right side of the quantization bin 0 (positive)
if isDC
    isUni = true;
else
    isUni = false;
end
for qBin = 1:1:qBinCenters(end)
    if ~isUni
        if qBin == 1 && qBin == qBinCenters(end)
            lambdaL = lambdaZero;
            borderL = pmfZero(end)*compTVqBinHist(ceil(end/2));
            borderR = 0;
        elseif qBin == 1
            lambdaL = lambdaZero;
            borderL = pmfZero(end)*compTVqBinHist(ceil(end/2));
            borderR = compTVqBinHist(ceil(end/2)+qBin+1)./Q;
        elseif qBin == qBinCenters(end)
            borderL = pmfPos(end)*compTVqBinHist(ceil(end/2)+qBin-1);
            borderR = 0;
        else
            borderL = pmfPos(end)*compTVqBinHist(ceil(end/2)+qBin-1);
            borderR = compTVqBinHist(ceil(end/2)+qBin+1)./Q;
        end
        
        if borderL <= borderR % uniform is better
            isUni = true; % the following bins: uniform is better
        else
            lambdas = [1e-3,0.005:0.005:lambdaL]; % search the threshold for lambda
            histPosMat = compTVqBinHist(ceil(end/2)+qBin).*get_pmf(Q,1,lambdas);
            if rem(Q,2) == 1 % Q is an odd number
                lambdasCands = lambdas(histPosMat(:,1) <= borderL & histPosMat(:,end) >= borderR);
            else % Q is an even number
                lambdasCands = lambdas(histPosMat(:,2) <= histPosMat(:,1)+borderL & histPosMat(:,end-1) >= histPosMat(:,end)+borderR/2);
            end
            lambdaMIN = min(lambdasCands); lambdaMAX = max(lambdasCands);            
            if isempty(lambdaMIN) || isempty(lambdaMAX) % uniform is better
                isUni = true; % the following bins: uniform is better
            else
                if lambdaMIN == lambdaMAX
                    lambdaPos = lambdaMIN;
                else
                    weights = 1./(abs(round(coefRange./Q) - qBin) + 1).^1;
                    lambdaPos = lsqcurvefit(func,(lambdaMIN+lambdaMAX)/2,coefRange,tvHist.*sqrt(weights)./sum(tvHist),lambdaMIN,lambdaMAX,optimset('Display','off'));
                end
                pmfPos = get_pmf(Q,1,lambdaPos);
                
                lambdaL = lambdaPos; % for the next round of searching the threshold for lambda
            end
        end
    end
    
    % add the dithering signal
    % (randomly, without considering the image spatial-domain information)
    yindex = find(round(compTVSubband./Q) == qBin);
    if isUni
        noises = get_dither(length(yindex),Q,1); % uniform distribution
    else
        noises = get_dither(length(yindex),Q,1,lambdaPos); % Laplacian distribution
    end
    ditheredCompTVSubband(yindex) = compTVSubband(yindex) + noises; % add the noise
end

%% quantization bins on the left side of the qantization bin 0 (negative)
if isDC
    isUni = true;
else
    isUni = false;
end
for qBin = -1:-1:qBinCenters(1)
    if ~isUni
        if qBin == -1 && qBin == qBinCenters(1)
            lambdaR = lambdaZero;
            borderL = 0;
            borderR = pmfZero(1)*compTVqBinHist(ceil(end/2));
        elseif qBin == -1
            lambdaR = lambdaZero;
            borderL = compTVqBinHist(ceil(end/2)+qBin-1)./Q;
            borderR = pmfZero(1)*compTVqBinHist(ceil(end/2));
        elseif qBin == qBinCenters(1)
            borderL = 0;
            borderR = pmfNeg(1)*compTVqBinHist(ceil(end/2)+qBin+1);
        else
            borderL = compTVqBinHist(ceil(end/2)+qBin-1)./Q;
            borderR = pmfNeg(1)*compTVqBinHist(ceil(end/2)+qBin+1);
        end
        
        if borderL >= borderR % uniform is better
            isUni = true; % the following bins: uniform is better
        else
            lambdas = [1e-3,0.005:0.005:lambdaR]; % search the threshold for lambda
            histNegMat = compTVqBinHist(ceil(end/2)+qBin).*get_pmf(Q,-1,lambdas);
            if rem(Q,2) == 1 % Q is an odd number
                lambdasCands = lambdas(histNegMat(:,1) >= borderL & histNegMat(:,end) <= borderR);
            else % Q is an even number
                lambdasCands = lambdas(histNegMat(:,2) >= histNegMat(:,1)+borderL/2 & histNegMat(:,end-1) <= histNegMat(:,end)+borderR);
            end
            lambdaMIN = min(lambdasCands); lambdaMAX = max(lambdasCands);
            if isempty(lambdaMIN) || isempty(lambdaMAX) % uniform is better
                isUni = true; % the following bins: uniform is better
            else
                if lambdaMIN == lambdaMAX
                    lambdaNeg = lambdaMIN;
                else
                    weights = 1./(abs(round(coefRange./Q) - qBin) + 1).^1;
                    lambdaNeg = lsqcurvefit(func,(lambdaMIN+lambdaMAX)/2,coefRange,tvHist.*sqrt(weights)./sum(tvHist),lambdaMIN,lambdaMAX,optimset('Display','off'));
                end
                pmfNeg = get_pmf(Q,-1,lambdaNeg);
                
                lambdaR = lambdaNeg; % for the next round of searching the threshold for lambda
            end
        end
    end
    
    % add the dithering signal
    % (randomly, without considering the image spatial-domain information)
    yindex = find(round(compTVSubband./Q) == qBin);
    if isUni
        noises = get_dither(length(yindex),Q,-1); % uniform distribution
    else
        noises = get_dither(length(yindex),Q,-1,lambdaNeg); % Laplacian distribution
    end
    ditheredCompTVSubband(yindex) = compTVSubband(yindex) + noises; % add the noise
end

% for debugging
% figure; bar(coefRange,hist(round(tvSubband),coefRange));
% figure; bar(coefRange,hist(round(ditheredCompTVSubband),coefRange));
% figure; bar(coefRange,hist(round(tvSubband./Q).*Q,coefRange));
% figure; bar(coefRange,hist(round(ditheredCompTVSubband./Q).*Q,coefRange));
% disp(max(abs(hist(round(tvSubband./Q).*Q,coefRange)-hist(round(ditheredCompTVSubband./Q).*Q,coefRange))));

end

function [pmf] = get_pmf(Q,flag,lambda)
% pmf of the distribution of the dithering signal
% dom(f) is the set integers between -Q/2 and Q/2
% note the difference when Q is an even number

if ~(size(lambda,1) >= 1 && size(lambda,2) == 1) % column vector
    lambda = lambda';
end
if ~(size(lambda,1) >= 1 && size(lambda,2) == 1)
    error('lambda should be a column vector!');
end
lN = length(lambda);

if rem(Q,2) == 1 % Q is an odd number
    pmf = zeros(lN,Q);
    if flag == 0 % center bin
        pmf(:,ceil(Q/2)) = (1-exp(-lambda./2)) ./ (1-exp(-lambda.*Q/2));
        pmf(:,ceil(Q/2)+1:Q) = exp(-repmat(lambda,[1,floor(Q/2)]).*repmat(1:floor(Q/2), [lN,1])) .* repmat(exp(lambda./2)-exp(-lambda./2),[1,floor(Q/2)]) ./ repmat(2*(1-exp(-lambda.*Q/2)),[1,floor(Q/2)]);
        pmf(:,1:ceil(Q/2)-1) = pmf(:,Q:-1:ceil(Q/2)+1);
    elseif flag > 0
        pmf = exp(-repmat(lambda,[1,Q]).*repmat((-floor(Q/2):floor(Q/2))+Q/2,[lN,1])) .* repmat(exp(lambda./2)-exp(-lambda./2),[1,Q]) ./ repmat(1-exp(-lambda.*Q),[1,Q]);
    else % flag < 0
        pmf = get_pmf(Q,-flag,lambda);
        pmf = pmf(:,end:-1:1); % symmetry
    end
else % Q is an even number
    pmf = zeros(lN,Q+1);    
    if flag == 0
        pmf(:,Q/2+1) = (1-exp(-lambda./2)) ./ (1-exp(-lambda.*Q/2));
        pmf(:,end) = -exp(-lambda.*Q/2) .* (1-exp(lambda./2)) ./ (2*(1-exp(-lambda.*Q/2)));
        pmf(:,Q/2+2:Q) = exp(-repmat(lambda,[1,Q/2-1]).*repmat(1:Q/2-1,[lN,1])) .* repmat(exp(lambda./2)-exp(-lambda./2),[1,Q/2-1]) ./ repmat(2*(1-exp(-lambda.*Q/2)),[1,Q/2-1]);
        pmf(:,1:Q/2) = pmf(:,Q+1:-1:Q/2+2);
    elseif flag > 0
        pmf(:,1) = (1-exp(-lambda./2)) ./ (1-exp(-lambda.*Q));
        pmf(:,end) = exp(-lambda.*Q) .* (exp(lambda./2)-1) ./ (1-exp(-lambda.*Q));
        pmf(:,2:end-1) = exp(-repmat(lambda,[1,Q-1]).*repmat((-Q/2+1:Q/2-1)+Q/2,[lN,1])) .* repmat(exp(lambda./2)-exp(-lambda./2),[1,Q-1]) ./ repmat(1-exp(-lambda.*Q),[1,Q-1]);
    else % flag < 0
        pmf = get_pmf(Q,-flag,lambda);
        pmf = pmf(:,end:-1:1); % symmetry
    end
end

end

function [d] = get_dither(N,Q,flag,lambda)
% generate the dithering signal

if ~(nargin == 4 && lambda > 0) % uniform distribution, for the DC component and maybe the tail of the AC components
    if flag == 0
        d = rand(1,N) - 0.5; % (-0.5,0.5)
        d = d.*Q;
    elseif flag == 1
        d = 2*abs(rand(1,N)-0.5) - 0.5; % [-0.5,0.5)
        d = d.*Q;
    elseif flag == -1
        d = -get_dither(N,Q,1); % symmetry
    else
        error('error - flag');
    end
else
    if flag == 0
        d = rand(1,N) - 0.5;
        d = sign(d).*(1/lambda).*log( (1-exp(-lambda*Q/2)).*(1-2*abs(d)) + exp(-lambda*Q/2) );
    elseif flag == 1
        d = 2*abs(rand(1,N)-0.5);
        d = -Q/2 - (1/lambda).*log( 1 - (1-exp(-lambda*Q)).*d );
    elseif flag == -1      
        d = -get_dither(N,Q,1,lambda); % symmetry
    else
        error('error - flag');
    end
end

end
