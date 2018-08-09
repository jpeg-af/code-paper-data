function [smthI] = PerDCThistSMTH(tvI,qtable,dimMAX)
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
% description:   perceptual DCT histogram smoothing
%
% INPUT
%           tvI: image matrix (pixel value) of the input image here the 
%                image is supposed to be the postprocessed JPEG image by a
%                process of TV-based JPEG anti-forensic method
%        qtable: quantization table, 8 * 8 sized
%        dimMAX: maximum dimension of the assignment for DCT histogram
%                smoothing, in order to reduce the computation cost
%
% OUTPUT
%         smthI: processed image with smoothed DCT histograms
%
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% Last modified: Aug. 6th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% process the DCT coefficients in zig-zag order
zigzagInds = [1 9 2 3 10 17 25 18 11 4 5 12 19 26 33 41 34 27 20 13 6 7 14 21 28 35 42 49 57 50 43 36 29 22 15 8 16 23 30 37 44 51 58 59 52 45 38 31 24 32 39 46 53 60 61 54 47 40 48 55 62 63 56 64];
ssimOffset = 8; % the offset for computing the SSIM value locally
borderTMAX = 2; % a threshold for distinguishing DCT histograms which can be ignored for DCT histogram smoothing

[nH,nW] = size(tvI); % image size
tvSubbands = im2vec(bdct(double(tvI)-128),8,0); % DCT coefficients to be processed

% the maximum dimension of the assignment problem
% when moving the DCT histogram towards the smoothed one
if ~(nargin == 3 && dimMAX > 0)
    % NOT to restrict the dimension of the assignment problem
    dimMAX = size(tvSubbands,2);
end

smthTVSubbands = tvSubbands; % initilization
smthI = tvI; % initilization

for subbandInd = zigzagInds % for each subband
    tic;
    
    % quantize DCT coefficients again with the same quantization step length
    Q = qtable(subbandInd); % quantization step length
    tvSubband = tvSubbands(subbandInd,:); % current subband
    smthTVSubband = tvSubband; % initilization
    coefMAX = max(abs(tvSubband));
    coefMAX = round(coefMAX/Q)*Q + (round(coefMAX/Q)<0)*floor(Q/2) + (round(coefMAX/Q)==0)*(ceil(Q/2)-1) + (round(coefMAX/Q)>0)*(ceil(Q/2)-1);
    coefRange = -coefMAX:coefMAX;
    
    % add adaptive dithering signal to the quantized DCT coefficients
    % as the reference for DCT histogram smoothing
    ditheredCompTVSubband = addRndAdaptDither(tvSubband,Q,subbandInd==1);
%     figure; bar(coefRange,hist(round(tvSubband./Q).*Q,coefRange)-hist(round(ditheredCompTVSubband./Q).*Q,coefRange));
%     tp1 = tvSubband(round(tvSubband./Q)==0); tp1 = hist(tp1,-floor(Q/2):floor(Q/2));
%     tp2 = ditheredCompTVSubband(round(ditheredCompTVSubband./Q)==0); tp2 = hist(tp2,-floor(Q/2):floor(Q/2));
%     tp1 = tvSubband(round(tvSubband./Q)==1); tp1 = hist(tp1,min(tp1):max(tp1));
%     tp2 = ditheredCompTVSubband(round(ditheredCompTVSubband./Q)==1); tp2 = hist(tp2,min(tp2):max(tp2));
%     tp1 = tvSubband(round(tvSubband./Q)==-1); tp1 = hist(tp1,min(tp1):max(tp1));
%     tp2 = ditheredCompTVSubband(round(ditheredCompTVSubband./Q)==-1); tp2 = hist(tp2,min(tp2):max(tp2));
    
    % sort the blocks
    % blocks with a relatively higher tolerance of distortion will be modified first
    if subbandInd == 1 % at first, smtI = tvI
        ssimInd = var(im2vec(smthI,8,0),1); % compute the variance instead
    else % using SSIM index
        [~,ssimInd] = ssim_index(tvI,smthI);
        ssimInd = sum(im2vec(PadOrCrop(ssimInd,size(tvI)),8,0),1);
    end
    
    % solve the simplified assignment problem for DCT histogram smoothing
    qBins = unique(round((coefRange)./Q)); % the quantization bins
    if numel(qBins) <= 1 % this kind of histogram can be left alone
        continue; % e.g. in some high-frequency subbands
    else
        borderT = max(sum(round(tvSubband./Q)==-1)./Q, sum(round(tvSubband./Q)==1)./Q); % say uniform distribution in the neighboring bins
        if subbandInd ~= 1 && borderT < borderTMAX % this kind of histogram can also be left alone
            continue; % e.g. in some high-frequency subbands
        else
            fprintf('Subband: %2d\tBin:  000000 - 000000 / [000000,000000]',subbandInd);
            
            for qBin = qBins(1):qBins(end) % within each quantization bin
                coefFromValues = tvSubband(round(tvSubband./Q)==qBin);
                if isempty(coefFromValues)
                    continue;
                end
                coefToValues = ditheredCompTVSubband(round(ditheredCompTVSubband./Q)==qBin);
                MAX = round(max(max(coefFromValues),max(coefToValues)));
                MIN = round(min(min(coefFromValues),min(coefToValues)));
                if MIN == MAX
                    continue;
                else
                    % constructing the histograms using integers as the bin
                    % centers and then subtracting them, in order to decide
                    % the DCT histogram moving strategy
                    diffNums = hist(round(coefFromValues),MIN:MAX) - hist(round(coefToValues),MIN:MAX);
                end
                fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%6d - %6d / [%6d,%6d]\t',qBin,sum(diffNums(diffNums>0)),qBins(1),qBins(end));
                
                % find the blocks with a relatively higher tolerance of
                % distortion to be changed later
                blockToChangeInds = [];
                blockMayChangeInds = round(tvSubband./Q)==qBin; % all the blocks whose DCT coefficients in this subband within this quantization bin
                for k = 1:length(diffNums)
                    if diffNums(k) > 0
                        ssimIndTP = ssimInd; ssimIndTP(~(blockMayChangeInds&round(tvSubband)==MIN+k-1)) = -1;
                        [~,sortInd] = sort(ssimIndTP,'descend');
                        blockToChangeInds = cat(2,blockToChangeInds,sortInd(1:diffNums(k)));
                    end
                end
                
                % find the targeted DCT coefficients values
                coefTargetValues = [];
                for k = 1:length(diffNums)
                    if diffNums(k) < 0
                        tp = coefToValues(round(coefToValues)==MIN+k-1);
                        coefTargetValues = cat(2,coefTargetValues,tp(1:-diffNums(k)));
                    end
                end
                coefTargetValues = coefTargetValues(randperm(numel(coefTargetValues))); % shuffle the coefficients to which the orginal coefficients will be changed
                
                % in order to reduce the computation cost,
                % randomly divide the assignment problem to several smaller
                % ones with maximum dimension $dimMAX$
                for asgInd = 1:ceil(numel(blockToChangeInds)/dimMAX)
                    if asgInd < ceil(numel(blockToChangeInds)/dimMAX)
                        blockToChangeInds0 = blockToChangeInds((asgInd-1)*dimMAX+1:asgInd*dimMAX);
                        coefTargetValues0 = coefTargetValues((asgInd-1)*dimMAX+1:asgInd*dimMAX);
                    else % last (small) assignment problem
                        blockToChangeInds0 = blockToChangeInds((asgInd-1)*dimMAX+1:end);
                        coefTargetValues0 = coefTargetValues((asgInd-1)*dimMAX+1:end);
                    end
                    coefTargetValues0 = sort(coefTargetValues0);
                    
                    % generate the cost matrix
                    costMat = zeros(numel(blockToChangeInds0),numel(coefTargetValues0));
                    for k = 1:length(blockToChangeInds0)
                        blkInd = blockToChangeInds0(k);
                        
                        % locate the local image patch for the current block
                        ssimOffset8 = ceil(ssimOffset/8)*8; % consider complete blocks first
                        [iRow,iCol] = ind2sub([nH/8,nW/8],blkInd); % block location
                        iRowOffsetMIN8 = (iRow==1)*0 - (iRow>1&iRow<nH/8)*ssimOffset8 - (iRow==nH/8)*2*ssimOffset8;
                        iRowOffsetMAX8 = (iRow==nH/8)*0 + (iRow<nH/8&iRow>1)*ssimOffset8 + (iRow==1)*2*ssimOffset8;
                        iColOffsetMIN8 = (iCol==1)*0 - (iCol>1&iCol<nW/8)*ssimOffset8 - (iCol==nW/8)*2*ssimOffset8;
                        iColOffsetMAX8 = (iCol==nW/8)*0 + (iCol<nW/8&iCol>1)*ssimOffset8 + (iCol==1)*2*ssimOffset8;
                        rows8 = (iRow-1)*8+1+iRowOffsetMIN8:iRow*8+iRowOffsetMAX8;
                        cols8 = (iCol-1)*8+1+iColOffsetMIN8:iCol*8+iColOffsetMAX8;
                        rowBlkInds = unique(ceil(rows8./8));
                        colBlkInds = unique(ceil(cols8./8));
                        [p,q] = meshgrid(rowBlkInds, colBlkInds);
                        blkInds = sort(sub2ind([nH/8,nW/8], p(:), q(:))); % locations of blocks (local image patch) under consideration
                        
                        % add the possible maximum noise
                        noiseMAX = max(abs(coefTargetValues0 - smthTVSubbands(subbandInd,blkInd)));
                        subbandsIndMAX = smthTVSubbands(:,blkInds);
                        subbandsIndMAX(subbandInd,blkInds==blkInd) = subbandsIndMAX(subbandInd,blkInds==blkInd) + noiseMAX;
                        noiseMAXIPatch = uint8(ibdct(vec2im(subbandsIndMAX,0,8,numel(rowBlkInds),numel(colBlkInds)))+128);
                        smthIPatch = smthI(rows8,cols8); % reference image patch
                        
                        % locate the image patch with the require pixel offset
                        if ssimOffset ~= ssimOffset8
                            if iRow == 1
                                rows = 1:8+2*ssimOffset;
                            else
                                rows = ssimOffset8-ssimOffset+1:8+ssimOffset+ssimOffset8;
                            end
                            
                            if iCol == 1
                                cols = 1:8+2*ssimOffset;
                            else
                                cols = ssimOffset8-ssimOffset+1:8+ssimOffset+ssimOffset8;
                            end
                            
                            ssimMAX = ssim(smthIPatch(rows,cols),noiseMAXIPatch(rows,cols));
                        else
                            ssimMAX = ssim(smthIPatch,noiseMAXIPatch);
                        end
                        
                        % using linear interpolation to estimate the possible distortions
                        ssims = interp1([noiseMAX,0],[ssimMAX,1],abs(coefTargetValues0-smthTVSubbands(subbandInd,blkInd)));
                        
                        % the cost for this DCT block
                        costMat(k,:) = 1 - ssims;
                        
                    end % end of for k = 1:length(blockToChangeInds0)
                    
                    % solve the assignment problem
                    sc = ceil(-log10(min(min(costMat(costMat~=0))))); % for scaling
                    if isempty(sc), sc = 0; end
                    assignment = munkres(round(costMat.*10.^sc)); 
                    
                    % modify the coefficients
                    smthTVSubband(blockToChangeInds0) = coefTargetValues0(assignment);
                    smthTVSubbands(subbandInd,:) = smthTVSubband;
                    
                    % transform the DCT coefficients back to the spatial domain
                    smthI = double(uint8(ibdct(vec2im(smthTVSubbands,0,8,floor(nH/8),floor(nW/8)))+128));
                    
                end % end of for asgInd = 1:ceil(numel(blockToChangeInds)/dimMAX)
                
            end % end of for bin = bins(1):bins(end)
            
        end % end of if subbandInd ~= 1 && borderT < 2
        
    end % end of numel(bins) <= 1
    
%     for debugging
%     close all;
%     figure; bar(coefRange,hist(round(tvSubband),coefRange));
%     figure; bar(coefRange,hist(round(ditheredCompTVSubband),coefRange))
%     figure; bar(coefRange,hist(round(smthTVSubband),coefRange));
%     disp(max(abs(hist(round(ditheredCompTVSubband),coefRange)-hist(round(smthTVSubband),coefRange))));
%     figure; bar(coefRange,hist(round(tvSubband./Q).*Q,coefRange));
%     figure; bar(coefRange,hist(round(ditheredCompTVSubband./Q).*Q,coefRange));
%     figure; bar(coefRange,hist(round(smthTVSubband./Q).*Q,coefRange));
%     disp(max(abs(hist(round(tvSubband./Q).*Q,coefRange)-hist(round(smthTVSubband./Q).*Q,coefRange))));
    
    toc;
end

end