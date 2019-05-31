function A = vec2fac(x,I)
R = numel(x)/sum(I);
A = mat2cell(x(:),I(:)*R,1);
A = cellfun(@(x) reshape(x,[],R),A,'uni',0);