function a = fac2vec(A)
a = cell2mat(cellfun(@(x) x(:),A(:),'uni',0));