%% Copyright (C) 2012 Telematic and Telecommunication Laboratory (LTT),       
% Dipartimento di Ingegneria dell'Informazione- Università di Siena
% via Roma 56 53100 - Siena, Italy                   
% 
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for removing statistical
%traces left into the histogram of an image by any processing operator. 
%Please refer to the following paper:
%
%M. Barni, M. Fontani, B. Tondi, "A Universal Technique to Hide Traces of 
%Histogram-Based Image Manipulations", ACM Int. Workshop on Multimedia and 
%Security (MMSEC) 2012, Coventry, UK, 2012.
%
%Kindly report any suggestions or corrections to marco.fontani@unisi.it
%
%----------------------------------------------------------------------
function[reflected] = PadOrCrop(img, target)
%function[reflected] = PadOrCrop(img, target)
%This function take the image (either RGB or grayscale) and adapt it to
%have size equal to that specified in target.
%If the target value for the image dimension is smaller than the actual,
%the algorithm crops the central part of the image;
%If the target value for the image dimension is bigger than the actual, the
%algorithm perform a reflective padding along that dimension.
%
%--- INPUT ARGS ---
%-> img: starting image (RGB or grayscale)
%-> target: 1-by-2 array of target dimensions
%
%--- OUTPUT ARGS ---
%-> reflected: adapted image

offset(2) = round((target(2)-size(img,2))/2);
offset(1) = round((target(1)-size(img,1))/2);
W = size(img,1);
H = size(img,2);
CH = size(img,3);

reflected = zeros(size(img,1)+2*offset(1) , size(img,2)+2*offset(2), CH);

%assert(prod(offset)>0,'[Error]: cannot perform mixed cropping and extension');

if offset(1)<0
    if offset(2)<0
        reflected = img(-offset(1)+1:end+offset(1),-offset(2)+1:end+offset(2),:);
        reflected = minor_fix(reflected,target);
        return;
    else
        reflected(:,offset(2)+1: H+offset(2),:) = img(-offset(1)+1:end+offset(1),:,:);  
        reflected(:,1:offset(2),:) = reflected(:, offset(2)*2:-1:offset(2)+1,:);
        reflected(:,end-offset(2)-1:end,:) = reflected(:,end-offset(2):-1:end-2*offset(2)-1,:,:);            
    end
else
    if offset(2)<0
        reflected(offset(1)+1: W+offset(1) , :, :) = img(:,-offset(2)+1:end+offset(2),:);
        reflected(1:offset(1),:,:) = reflected(offset(1)*2:-1:offset(1)+1,:,:);    
        reflected(end-offset(1)-1:end,:,:) = reflected(end-offset(1):-1:end-2*offset(1)-1,:,:);        
    else
        reflected(offset(1)+1: W+offset(1) , offset(2)+1: H+offset(2), :) = img;
        reflected(1:offset(1),:,:) = reflected(offset(1)*2:-1:offset(1)+1,:,:);    
        reflected(end-offset(1)-1:end,:,:) = reflected(end-offset(1):-1:end-2*offset(1)-1,:,:);
        reflected(:,1:offset(2),:) = reflected(:, offset(2)*2:-1:offset(2)+1,:);
        reflected(:,end-offset(2)-1:end,:) = reflected(:,end-offset(2):-1:end-2*offset(2)-1,:,:);            
    end
end
    reflected = minor_fix(reflected,target);
    return;
end

function[fixed] = minor_fix(array, trgdim)
    if size(array,1)>trgdim(1)
        array = array(1:end-1,:,:);
    elseif size(array,1)<trgdim(1)
        array(end+1,:,:) = array(end,:,:);
    end   
    
    if size(array,2)>trgdim(2)
        array = array(:,1:end-1,:);
    elseif size(array,2)<trgdim(2)
        array(:,end+1,:) = array(:,end,:);
    end  
    fixed=array;
end