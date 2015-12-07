function k = curvegram(x,y, nscales_block, n_blocks, frmin, frmax); 
% CURVEGRAM
%  
%  k = curvegram(x,y, nscales_block, n_blocks, frmin, frmax); 
%  
%  CURVEGRAM: Multiscale curvature, with the algorithm based on Costa & Cesar,
%          Shape Classification and Analysis, CRC Press, 2000.
%
%  x,y: x and y coordinates of the 2D parametric curve
%
%  nscales_block, n_blocks: The multiscale curvature is calculated for a
%  number of different scales. The total number of scales is
%  nscales_block * n_blocks (e.g. for nscales_block=2 n_blocks=3, there
%  are 3 blocks of 2 scales each, totalizing 2*3=6 different scales). This
%  solution of blocks of scales is adopted because of memory
%  limitations in Matlab.
%
%  frmin, frmax: the scale is the inverse of frequency, ie
%  frequency = 1/scale. frmin and frmax are the minimum and the maximum
%  frequencies, which define the minimum and the maximum scales
%  calculated for the multiscale curvature (i.e. curvegram).
%
% 
%  See also 
% 
%  Further Information:  
%       http://www.ime.usp.br/~cesar  
%  
%   Copyright (c) 2001 by Roberto M Cesar Jr. 
%   e-mail:cesar@ime.usp.br  
%   $Revision: 0.0 $  $Date: 2001/01/16 $ 

% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% 
% %                                                                   % %  
% %  curveg5.m:               
% %                 
% %                                                                   % %  
% % Feito por Roberto Marcondes Cesar Junior                          % %  
% % e-mail : pinda@.ifsc.sc.usp.br                                   % %  
% %                                                                   % %  
% % data :  11/06/97       
% %                                                                   % %  
% % observacoes : Gaussiana do Papoulis; escala exponential 
% %               Otimizado para nao usar lacos! 
% %               Tambem para dividir as escalas e nao depender do disco 
% %               MORPHOGRAM-BASED SCALE SET-UP                       % %  
% %               GAUSS=EXP(-S^2/2*SG^2)                                                    % %  
% %  
% %                                                                   % %  
% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%  


if (nargin<3)
 dk = 1;
else
 dk = nscales_block;
end

if (nargin<4)
 Nk = 1;
else
 Nk = n_blocks;
end

SigMax = Nk*dk; 

Np = length(x);
Np1 = Np - 1; 
Np2 = floor(Np/2); 

if (nargin<6)
 frmin = log(Np)/2;
 frmax = Np/2;
end


it = (0:(Nk-1))'*ones(1,dk); 
jt = ones(Nk,1)*(1:dk); 
sgk = it * dk + jt; 
 
clear it jt 
 
u = (x + j*y).'; 
 
logmin = log(frmin); 
logmax = log(frmax); 

if (SigMax>1)
 sgb = exp(((logmax - logmin)/(SigMax - 1)) * (sgk - 1) + logmin); 
else
 sgb(1,1) = frmin;
end

U0 = fft(u); 
 
% tira o DC 
U0(1) = 0; 
 
% u sem DC 
tu = ifft(U0); 
 
U0 = fftshift(U0); 
 
E0 = sum( abs(U0) ); 
 
s = -Np2:(Np1-Np2); 
 
dU0 = j * U0 .* s; 
 
du0 = ifft(fftunsh(dU0)); 
 
p0 = (2*pi/Np) * sum( abs(du0) ); 
cols = 1:Np; 
ru = zeros(SigMax,Np); 
du = zeros(SigMax,Np); 
ddu = zeros(SigMax,Np); 
sm = ones(dk,1) * s; 
clear dU0 du0 
 
for k=1:Nk 
 fprintf('%d ',k); 
 
 sg = sgb(k,1:dk);  
 
 U = ones(dk,1) * U0; 
 
 % banco de filtros gaussianos 
 GF = exp( - (s.^2).' * (2*(sg).^(-2))).' ; 
 
 U = U .* GF; 
 dU = j * sm .* U; 
 ddU = (j*sm).^2 .* U; 
 
 ru(((k-1)*dk+1):(k*dk),cols) = ifft(fftsh1d(U).').'; 
 
 du(((k-1)*dk+1):(k*dk),cols) = ifft(fftsh1d(dU).').'; 
 
 ddu(((k-1)*dk+1):(k*dk),cols) = ifft(fftsh1d(ddU).').'; 
 
 clear U dU ddU GF 
 
end  

clear U0  sm GF
 
for i=1:SigMax 
 Cp(i) = p0 / ((2*pi/Np)*sum( abs(du(i,1:Np)))); 
end 
 
% NORMALIZACAO PELO PERIMETRO 
Cn = Cp; 
 
ru = ru .* (ones(Np,1) * Cn).'; 
du = du .* (ones(Np,1) * Cn).'; 
ddu = ddu .* (ones(Np,1) * Cn).'; 
 
% Complex Curvature 
 
k = (-imag( du .* conj(ddu)))./ (abs(du).^3); 
 
% flipud so that scale increases downwards
k = flipud(k); 
