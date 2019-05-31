function vargout = call_prox_nuclearnorm(mode,varargin)
persistent curr_nuclear_norm curr_x;

if mode == 1
    % compute prox_nuclearnom
    [x,infos] = prox_nuclearnorm(varargin{1},varargin{2},varargin{3});
    vargout = x; 
    curr_x = x;
    curr_nuclear_norm = infos.final_eval;
elseif mode == 2
    % return nuclear norm 
    x = varargin{1};
    if isempty(curr_x) || any(size(curr_x)~=size(x)) || norm(curr_x(:) - x(:)) > 1e-8
        curr_nuclear_norm = norm_nuclear(x);
    end
    vargout = curr_nuclear_norm; 
end
end