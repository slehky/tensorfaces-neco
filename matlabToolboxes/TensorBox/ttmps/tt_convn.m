function z = tt_convn(x,y)
% Convolution between two TT-tensors
% Phan Anh Huy, 2016.
%

%%
szx = size(x);
szy = size(y);
rx = rank(x);
ry = rank(y);

d = ndims(x);
Zc = cell(d,1);
for k = 1:d
    % METH 1
%     temp1 = nkron(x{k},ones(ry(k),1,ry(k+1)));  
%     temp2 = nkron(ones(rx(k),1,rx(k+1)),y{k});
%        
%     temp1x = cat(2,zeros(rx(k)*ry(k),szy(k),rx(k+1)*ry(k+1)),temp1);
%     temp2x = cat(2,zeros(rx(k)*ry(k),szx(k),rx(k+1)*ry(k+1)),temp2);
%     temp3 = ifft(fft(temp1x,[],2).*fft(temp2x,[],2),[],2);
%     temp3 = temp3(:,1:end-1,:);
    
    % METH2    
    temp1x = cat(2,zeros(rx(k),szy(k),rx(k+1)),x{k});
    temp2x = cat(2,zeros(ry(k),szx(k),ry(k+1)),y{k});
    
    temp1x = fft(temp1x,[],2);
    temp2x = fft(temp2x,[],2);
    
    temp1x = reshape(permute(temp1x,[1 3 2]),[],szy(k)+szx(k));
    temp2x = reshape(permute(temp2x,[1 3 2]),[],szy(k)+szx(k));
    
    temp3 = khatrirao(temp2x,temp1x);
    temp3 = reshape(temp3,rx(k),rx(k+1),ry(k),ry(k+1),[]);
    temp3 = permute(temp3,[1 3 5 2 4]);
    temp3 = reshape(temp3,rx(k)*ry(k),[],rx(k+1)*ry(k+1));
    
    temp3 = ifft(temp3,[],2);
    temp3 = temp3(:,1:end-1,:);
    
    Zc{k} = temp3;
end
z = cell2core(tt_tensor,Zc(:)');