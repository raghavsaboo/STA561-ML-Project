function u1 = chainpoint(u0,cc); 
%CHAINPOINT
%  
%  u1 = chainpoint(u0,cc);
% 
%  
%  Returns the neighbor in the CC chain-code direction. 
%
%  See also 
% 
%  Further Information:  
%   http://www.ime.usp.br/~cesar/projects/matlab/mulstiscale  
%  
%  This program is free software; you can redistribute it and/or
%  modify it under the terms of the GNU General Public License
%  as published by the Free Software Foundation; either version 2
%  of the License, or (at your option) any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software
%  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
%
%   Copyright (c) 2001 by Roberto M Cesar Jr. 
%   e-mail:cesar@ime.usp.br  
%   $Revision: 0.0 $  $Date: 2001/01/01 $ 

ccfactor = [1; 1-j; -j; -1-j; -1; -1+j; j; 1+j; 1]; 
 
u1 = u0 + ccfactor( cc+1 ); 
 