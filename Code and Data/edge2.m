function [OUT] = edge2(IN) 
% EDGE2 finds edges in a binary image. 
% IN - logic array (binary image). 
% OUT - logic array (binary image) of the same size as IN. 
% 
% Binary image IN: 
% (0) "false" - background, 
% (1) "true" - objects. 
% Binary image OUT: 
% (1) "true" where the function finds edges, 
% (0) "false" elsewhere. 
% 
% Remarks 
% ------- 
% Edges can be internal and external. This function finds the EXTERNAL 
% edges. But if you need to find the internal edges, you must invert 
% a binary image IN. 
% 
% Example 
% ------- 
% Find edges in the binary image IN: 
% 
% OUT = edge2(IN); external edges 
% OUT = edge2(~IN); internal edges 
% 
% See also: EDGE 
% 
% Author: Nazar Petruk 
% Address: TSTU, Ukraine 
% email: petruk.n.p@gmail.com 
% Date: 2013, April 
% Copyright (c) 2013, by Nazar Petruk

[r,c] = size(IN); 
% Creating a logic array A. The dimension of the array A is equal to 
% the dimension of the array IN plus 2(two). 
A = false([r,c]+2); 
% Fill the array A, using the operation "OR". 
for i = 0:2, 
    for j = 0:2, 
        A((1:r)+i,(1:c)+j) = A((1:r)+i,(1:c)+j) | IN; 
    end; 
end; 
% Receiving array OUT, using the operation "exclusive-OR". 
OUT = xor(A(2:r+1,2:c+1), IN);