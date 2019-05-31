function Y = mttkrpmem(Y,A,side,memfree)
if isempty(memfree)
    memfree = getfreemem;%memfree = 1920;
end
nmY = numel(Y);
I = cell2mat(cellfun(@size,A(:),'uni',0));
R = I(1,2);I = I(:,1)';

if strcmp(side,'right')
    facind = 1:numel(A);

    nkrp = find(memfree>=(8*R*(cumprod(I(facind)) + ...
        nmY./cumprod(I(facind)))),1,'last');
    % nkrp = nkrp-1;
    
    
    KRP = fkhatrirao(A(facind(1:nkrp)));
    
    Y = reshape(Y,[],prod(I(facind(1:nkrp))));
    Y = Y * KRP;
    nmY = nmY/prod(I(facind(1:nkrp)))*R;
    facind = facind(nkrp+1:end);
    
    while ~isempty(facind)
        clear KRP;
        nkrp = find(memfree >=(8*(R*cumprod(I(facind)) + ...
            nmY./cumprod(I(facind)))),1,'last');
        
        KRP = fkhatrirao(A(facind(1:nkrp)));
        Y = reshape(Y,[],prod(I(facind(1:nkrp))),R);
        Y = bsxfun(@times,Y,reshape(KRP,1,[],R));
        Y = squeeze(sum(Y,2));
        nmY = nmY/prod(I(facind(1:nkrp)));
        facind = facind(nkrp+1:end);
    end
    
elseif strcmp(side,'left')
    facind = numel(A):-1:1;
    nkrp = find(memfree>=(8*R*(cumprod(I(facind)) + ...
        nmY./cumprod(I(facind)))),1,'last');
    % nkrp = nkrp-1;
    
    
    KRP = fkhatrirao(A(facind(nkrp:-1:1)));
    
    Y = reshape(Y,prod(I(facind(nkrp:-1:1))),[]);
    Y = KRP' * Y;
    nmY = nmY/prod(I(facind(nkrp:-1:1)))*R;
    facind = facind(nkrp+1:end);
    
    while ~isempty(facind)
        clear KRP;
        nkrp = find(memfree >=(8*(R*cumprod(I(facind)) + ...
            nmY./cumprod(I(facind)))),1,'last');
        
        KRP = fkhatrirao(A(facind(nkrp:-1:1)));
        Y = reshape(Y,R,prod(I(facind(nkrp:-1:1))),[]);
        Y = bsxfun(@times,Y,KRP');
        Y = squeeze(sum(Y,2));
        nmY = nmY/prod(I(facind(nkrp:-1:1)));
        facind = facind(nkrp+1:end);
    end
    
end

