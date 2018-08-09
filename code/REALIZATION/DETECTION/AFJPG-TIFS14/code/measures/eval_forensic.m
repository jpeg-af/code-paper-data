clear;clc;

genDir='E:\anti-forensics\week3\forensic_method\networkL8\8.0restore\'
addpath('E:\anti-forensics\week3\forensic_method\AFJPG-TIFS14\code\jpegtbx_1.4')
addpath('E:\anti-forensics\week3\forensic_method\AFJPG-TIFS14\code\measures')
% im_names=dir(strcat(genDir,'*.png'));
im_names=dir(strcat(genDir,'*.png'));
num=length(im_names)
[cf_sum, bm_sum,bgm1_sum,bgm2_sum]=deal(0)

fprintf('cali_feature | blk_measure | blk_grad_measure1 | blk_grad_measure2 | qtable_est\n')
for i=1:num
    disp(i);
    im_name=num2str(im_names(i).name);
    g=imread(strcat(genDir,im_names(i).name));
    qtbl_est=qtable_est(g);
    blk_sig=blk_measure(g);
    
    fprintf('%10.4f %10.4f %10.4f %10.4f %10.4f\n',cali_feature(g),blk_measure(g),blk_grad_measure(g,1),blk_grad_measure(g,2),sum(sum(qtable_est(g)>1)));
    cf_sum=cali_feature(g)+cf_sum;
    bm_sum=blk_measure(g)+bm_sum;
    bgm1_sum=blk_grad_measure(g,1)+bgm1_sum;
    bgm2_sum=blk_grad_measure(g,2)+bgm2_sum;
end
% avg_fun=@(x) x/num
% [cf_avg, bm_avg,bgm1_avg,bgm2_avg]=feval(avg_fun,[cf_sum, bm_sum,bgm1_sum,bgm2_sum])
cf_avg=cf_sum/num;
bm_avg=bm_sum/num;
bgm1_avg=bgm1_sum/num;
bgm2_avg=bgm2_sum/num;
fprintf('average values of  metrics:\n');
fprintf('cali_feature | blk_measure | blk_grad_measure1 | blk_grad_measure2\n');
fprintf('%10.4f %10.4f %10.4f %10.4f\n',cf_avg, bm_avg,bgm1_avg,bgm2_avg);