function A = transpose(A)
U = A.U;
U = cellfun(@(x) permute(x,[1 3 2 4]),U,'uni',0);
A.U = U;
A = update_properties(A);
end