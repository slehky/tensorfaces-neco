function x = TTeMPS_op_rand(r,n,m,constraints)
%TTeMPS_op_rand Generates a random TTeMPS matrix
%   X = TTEMPS_RAND( R, N ) creates a length(N)-dimensional TTeMPS tensor
%   of size N(1)*N(2)*...N(end) with ranks R by filling the the cores with
%   uniform random numbers.
%  
%
% An orthogonal matrix should have the following format 
%
%        I1          I2          I3
%        |           |           |    
%   1--(U 1)-- R2--(U 2)-- R3--(U 3)-- 1
%        |           |           |    
%        1           1           J    
% where J < R3/I3
%
%        I1          I2          I3
%        |           |           |    
%   1--(U 1)-- R2--(U 2)-- R3--(U 3)-- 1
%        |           |           |    
%        1           J           1
%
%
% where J < R2*R3/I2


if nargin < 4
    constraints = 'random';
end
    
if length(r) ~= length(n)+1
    error('Size mismatch in arguments')
end

U = cell(1, length(n));
for i=1:length(n)
    sz = [r(i), n(i) m(i), r(i+1)];
    
    switch constraints
        case 'orthogonal'
            z1 = r(i)*n(i);
            z2 = m(i)*r(i+1);
            alft = orth(rand(z1, min(z1,z2)));
            arght = orth(rand(z2, min(z1,z2)));
            U{i} = alft * arght';
            U{i} = reshape(U{i},sz);
            
            
        case 'nonnegative'
            U{i} = rand(sz);
            
        otherwise
            U{i} = randn(sz);
            U{i} = U{i}/norm(U{i}(:));
    end
end
x = TTeMPS_op(U);
end
