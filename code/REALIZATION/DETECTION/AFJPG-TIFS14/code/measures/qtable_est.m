function [qtable] = qtable_est(im)
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
% description:   quantization table estimation using the MLE
% 
% requires:      Matlab JPEG toolbox
% 
% INPUT
%            im: image matrix containing pixel values
% 
% OUTPUT
%        qtable: estimated quantization table
% 
% reference:     Z. Fan and R. L. De Queiroz, "Identification of bitmap 
%                compression history: JPEG detection and quantizer
%                estimation," IEEE Trans. Image Process., vol. 12, no. 2, 
%                pp. 230–235, 2003.
% 
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Nov. 12th, 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 8; % DCT block size
[nH,nW] = size(im); % image size
im = double(uint8(im)); % ensure pixel values are integers with in [0,255]
im = im(1:n*floor(nH/n),1:n*floor(nW/n)); % proper cropping

%% initilization
qtable = ones(8,8);

%% exclude blocks which won't be used in the further estimation
imv = im2vec(im,8,0); % stacking
imv(:,sum(imv==0|imv==255,1) > 0) = []; % remove truncated blocks
imv(:,max(imv)==min(imv)) = []; % remove uniform blocks

%% compute DCT coefficients for the blocks in estimation
dctm = bdctmtx(8);
subbands = round(dctm*imv);
N = size(subbands,2); % the number of blocks

%% bounds
B = get_B();

%% 64 subbands
for k = 1:64
    subband = subbands(k,:);
    pfreq = hist(subband, min(subband):max(subband));
    if k==1 % DC component
        [~,indcenter] = max(pfreq); % center of the main lobe
        subband = subband - (min(subband)+indcenter-1);
        pfreq = hist(subband, min(subband):max(subband));
    end
    
    %% search possible quantization step lengths
    [~,ind0] = max(pfreq); % center of the main lobe
    [~,ind] = max(pfreq(ind0+floor(B(k))+2:end)); % Y > B
    if isempty(ind) % "undetermined"
        qtable(k) = 0;
        continue;
    end
    Q = ind + floor(B(k));
    Qs = unique([int_fracs(Q-1),int_fracs(Q),int_fracs(Q+1)]);
    
    %% test each possible quantization step length
    mlls = zeros(length(Qs),1); % maximum log-likelihood
    for l = 1:length(Qs)
        mlls(l) = logl(subband,Qs(l),B(k)) + N*log(Qs(l));
    end
    
    %% the Q maximizing the likelihood term
    [~,indML] = max(mlls);
    qtable(k) = Qs(indML);
end

end

function [B] = get_B()
%% bounds

B = zeros(8,8);
for k = 1:64
    [ii,jj] = ind2sub([8 8], k);
    B(k) = get_D(ii)*get_D(jj);
end

end

function [D] = get_D(n)

if n == 0+1 || n == 4+1
    D = 2;
elseif n == 2+1 || n == 6+1
    D = 2*cos(pi/4);
else
    D = 2*cos(pi/4)*cos(pi/8);
end

end

function [intfs] = int_fracs(n)
%% integer fractions

intfs = 1:n;
intfs = intfs(rem(n,1:n) == 0);

end

function [ml] = logl(subband,Q,B)

INF_neg = -100;

%% frequency histogram
pfreq = hist(subband, min(subband):max(subband));

%% -q/2 <= i <= q/2
is = -floor(Q/2):floor(Q/2);

%% N(i): the number of blocks satisfying Y-qr == i
Ns = zeros(size(is));
for k = 1:length(Ns)
    ds = [min(subband):max(subband)]-Q*round([min(subband):max(subband)]/Q);
    Ns(k) = sum(pfreq(ds == is(k)));
end

%% W(i)
Ws = zeros(size(is));    
mu = 0; sigma = sqrt(1/12); % alpha = sqrt(1/6);
for ii = 0:1:is(end)
    indI = ii-is(1)+1;
    js = ceil(-(B-0.5-abs(ii))/Q):floor((B-0.5-abs(ii))/Q);

    if ~isempty(js)
        ws = zeros(size(js));
        for jj = js(1):1:js(end)
            indJ = jj-js(1)+1;            
            g_inf = ii+jj*Q-0.5; g_sup = ii+jj*Q+0.5;
                    
%             if abs(g_inf)<=B && abs(g_sup)<=B
                ws(indJ) = normcdf(g_sup,mu,sigma) - normcdf(g_inf,mu,sigma);
%             elseif (g_inf<-B) && abs(g_sup)<=B
%                 fprintf('exceeded left border: %d\t%d\t%d\t%d\n',Q,B,ii,jj);
%                 ws(indJ) = normcdf(g_sup,mu,sigma) - normcdf(-B,mu,sigma);
%             elseif abs(g_inf)<=B && (g_sup>B)
%                 fprintf('exceeded right border: %d\t%d\t%d\t%d\n',Q,B,ii,jj);
%                 ws(indJ) = normcdf(B,mu,sigma) - normcdf(g_inf,mu,sigma);
%             end

% % %             ws(indJ) = cdf_ggd(g_sup/alpha,2) - cdf_ggd(g_inf/alpha,2);

        end
        
        ws = ws.*(sigma*sqrt(2*pi));
        if sum(ws) ~= 0
            Ws(indI) = log(sum(ws));
        else
            Ws(indI) = INF_neg;
        end
    else
        Ws(indI) = INF_neg;
    end
end
Ws(1:-is(1)) = Ws(end:-1:2-is(1)); % even function

%% the log-likehood term
ml = sum(Ns.*Ws);

end
