function z= cvx_sqp_diag(eg,F,delta)
% min  x'*diag(s) * x - 2*F'*x;
R= numel(eg);

eg2 = sqrt(eg);
b = F./eg2;

% cvx_begin
% cvx_precision best
% variable z(R,1)
% minimize sum_square(z.*eg2 - b)
% subject to
% norm(z) <= delta
% cvx_end


% 
% b = F(:);Q =diag(eg);
% cvx_begin
% cvx_precision best
% variable z(R,1)
% minimize   (quad_form(z,Q)-2*b'*z)
% subject to
% norm(z) <= delta
% cvx_end

% b = F(:); 
% cvx_begin
% cvx_precision best
% variable z(R,1)
% minimize   (eg.'*(z.^2)-2*b'*z)
% subject to
% norm(z) <= delta
% cvx_end

b = F(:); 
cvx_begin
cvx_precision best
variable z(R,1)
minimize   ((eg.*z-2*b)'*z)
subject to
norm(z) <= delta
cvx_end

% 
% b = F(:);
% eg = eg/norm(b);
% % eg = eg-eg(1) + 1;
% Q =diag(eg);
% cvx_begin
% cvx_precision best
% variable z(R,1)
% minimize   (quad_form(z,Q)-2*b'*z)
% subject to
% norm(z) <= delta
% cvx_end
% 
% 
% %