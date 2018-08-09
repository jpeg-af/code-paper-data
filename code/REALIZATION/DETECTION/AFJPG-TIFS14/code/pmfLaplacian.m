function [ydata] = pmfLaplacian(lambda,xdata,weights)
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
% dom(f) is the set of integers
% i.e. elements of $xdata$ are integers
% $weights$ is for weighted fitting
% contact: wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: Jul. 31st, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ydata = zeros(size(xdata)); % initialization
ydata(xdata~=0) = exp(-lambda*abs(xdata(xdata~=0)))*sinh(0.5*lambda); % x != 0
ydata(xdata==0) = 1 - exp(-0.5*lambda); % x = 0

ydata = ydata.*weights;

end