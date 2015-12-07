function [x,y] = contfol(img); 
%CONTFOL
%  
%  [x,y] = contfol(img);
%  
%  CONTFOL: contour following, with the algorithm based on Costa & Cesar,
%          Shape Classification and Analysis, CRC Press, 2000.
% 
%  See also 
% 
%  Further Information:  
%       http://www.ime.usp.br/~cesar  
%  
%   Copyright (c) 2001 by Roberto M Cesar Jr. 
%   e-mail:cesar@ime.usp.br  
%   $Revision: 0.0 $  $Date: 2001/01/16 $ 


 
% recebe uma imagem binaria: 0 para fundo e 1 para objeto 
 
[lins,cols] = size(img); 
 
 
% first object element in scanline 
 
[c0,l0] = find(img'); 
 
c0 = c0(1)-1; 
l0 = l0(1); 
 
% complex signal contour 
 
u(1) = c0 + j*l0; 
 
% number of points 
n=2; 
 
% find the second contour point 
found=0; 
cc = 8; % chain-code 
 
while (~found) 
  up = chainpoint(u(1),cc-4); 
  upn =  chainpoint(u(1),mod(cc+1,8)); 
  if (~img(imag(up),real(up)) & img(imag(upn),real(upn))) 
    found=1; 
  else 
    cc = cc+1; 
  end 
end 
 
next_pixel = up; 
dcn = cc; 
 
while (next_pixel ~= u(1)) 
  u(n) = next_pixel; 
  dpc = dcn; 
  [next_pixel,dcn] = find_next(u(n),dpc,img); 
  n = n+1; 
end 

np = length(u);

% INVERT TO BE COUNTERCLOCKWISE FROM MATRIX IJ COORDINATES
x = real(u(np:-1:1));
y = imag(u(np:-1:1));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUXILIARY FUNCTION 
 
function [pn, dcn] = find_next(pc, dpc, img) 
 
dcp = mod(dpc+4,8); % invert 
for r=0:6 
  de = mod(dcp+r,8); 
  dint = mod(dcp+r+1,8); 
  pe = chainpoint(pc,de); 
  pint = chainpoint(pc,dint); 
  if (~img(imag(pe),real(pe)) & img(imag(pint),real(pint))) 
    pn = pe; 
    dcn = de; 
 end 
end 
