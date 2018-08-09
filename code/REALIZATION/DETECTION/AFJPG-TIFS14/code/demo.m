% demo.m
% 
% Please be patient, it may take 3-10 minutes to create a JPEG forgery from 
% a JPEG image of size around 512*512. The computation cost also depends on 
% the quality factor of the JPEG image and the image resolution.

clear;

addpath('./measures/');
addpath('./jpegtbx_1.4/');

%% parameter setting
dimMAX = 200; % maximum dimension of the simplified assignment problem
lambda1 = 1.5; iterN1 = 50; steps1 = 1./(1:iterN1); % for the 1st round deblocking
lambda2 = 0.9; iterN2 = 30; steps2 = 1./(2:iterN2+1); % for the 2nd round deblocking
blkT = 0.0750; caliT = 0.0533; % example thresholds for 'blk_measure' and 'cali_feature' outputs

%% image information
imgname = 'lena.pgm'; q = 50;
jpgname = [imgname(1:length(imgname)-4),'-',num2str(q,'%.2d'),'.jpg'];
fprintf('\n\n%s\n',jpgname);
dbkname = [jpgname(1:length(jpgname)-4),'_tvdbk.pgm'];
dbkhsmthname = [jpgname(1:length(jpgname)-4),'_tvdbkHSMTH.pgm'];
forgname = [jpgname(1:length(jpgname)-4),'_forg.pgm'];

%% read the image
I = double(imread(imgname));
jpgI = double(imread(jpgname));
jobj = jpeg_read(jpgname);
Q = jobj.quant_tables{1};
dctQCoefs = dequantize(jobj.coef_arrays{1},Q);
fprintf('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n',psnr(jpgI,I),ssim(jpgI,I),cali_feature(jpgI),blk_measure(jpgI),blk_grad_measure(jpgI,1),blk_grad_measure(jpgI,2),sum(sum(qtable_est(jpgI)>1)));

%% the 1st round TV-based deblocking
fprintf('\n');
tic;
projFun = @(I) randPOQCS(I,dctQCoefs,Q,false);
tvI = TVsubGradJPEGDeblk(projFun,jpgI,lambda1,iterN1,steps1);
imwrite(uint8(tvI),dbkname);
toc;
fprintf('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n',psnr(tvI,I),ssim(tvI,I),cali_feature(tvI),blk_measure(tvI),blk_grad_measure(tvI,1),blk_grad_measure(tvI,2),sum(sum(qtable_est(tvI)>1)));

%% perceptual DCT histogram smoothing
fprintf('\n');
tvhsmthI = PerDCThistSMTH(tvI,Q,dimMAX);
imwrite(uint8(tvhsmthI),dbkhsmthname);
fprintf('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n',psnr(tvhsmthI,I),ssim(tvhsmthI,I),cali_feature(tvhsmthI),blk_measure(tvhsmthI),blk_grad_measure(tvhsmthI,1),blk_grad_measure(tvhsmthI,2),sum(sum(qtable_est(tvhsmthI)>1)));

%% the 2nd round TV-based deblocking + de-calibration
fprintf('\n');
tic;
projFun = @(I) randPOQCS(I,dctQCoefs,Q,true);
tvhsmthtvI = TVsubGradJPEGDeblk(projFun,tvhsmthI,lambda2,iterN2,steps2,blkT,true);
fprintf('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n',psnr(tvhsmthtvI,I),ssim(tvhsmthtvI,I),cali_feature(tvhsmthtvI),blk_measure(tvhsmthtvI),blk_grad_measure(tvhsmthtvI,1),blk_grad_measure(tvhsmthtvI,2),sum(sum(qtable_est(tvhsmthtvI)>1)));
toc;

tic;
fprintf('\n');
tvhsmthtvcaliI = deCalibration(tvhsmthtvI,caliT);
fprintf('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n',psnr(tvhsmthtvcaliI,I),ssim(tvhsmthtvcaliI,I),cali_feature(tvhsmthtvcaliI),blk_measure(tvhsmthtvcaliI),blk_grad_measure(tvhsmthtvcaliI,1),blk_grad_measure(tvhsmthtvcaliI,2),sum(sum(qtable_est(tvhsmthtvcaliI)>1)));
toc;

imwrite(uint8(tvhsmthtvcaliI),forgname); % write the forgey into an image file
