function makeTensorFaces(numcomponents,rankValue,facesfile)
% FUNCTION MAKETENSORFACES(NUMCOMPONENTS,RANKVALUE,FACEINPUT)
% Does tensor decomposition and stores results to a file.

% numfactors: number of components in the tensor decomposition
% rankValue: rank of components in the tensor decomposition
% facesfile: name of file containing faces
%
% Example: makeTensorFaces(20,8,'trainingset128')
%
% Output is saved to a file. The file contains the variables:
% 1. components
% 2. weights
%
% Sidney Lehky 2019

% Edit to specify your "homedirectory" in file "setupDirectories"
setupDirectories

% Load array of faces, which are in RGB color space
% The variable being loaded is "face"
eval(['load ',char(facesfile)]);

sizefaces = size(face);  % size of array holding the face set

% convert color spaces, RGB faces to LAB faces; LAB color space is more
% physiological
facearray = rgb2lab(face); clear face

%sizefaces = size(facearray);
imagerows = size(facearray,1);  % number of pixel rows in face image
imagecolumns = size(facearray,2);  % number of pixel columns in face image
numfaces = size(facearray,4);

L = [rankValue*ones(1,numcomponents)];   % rank of patterns A_p, L = [L1, L2,..., LP]
M = [ones(1,numcomponents)];    % rank of patterns X_p, M = [M1, M2,..., MP]

% L: Rank of basis, specifies complexity of decomposition
% M: Rank of coefficients. Always set to 1.0 for this decomposition
% In principal one can specify L values individually for each
% component, but here we set the arrays constant for all components.

% derived parameters
LM = [L;M];  % just concatenates L and M
facearrayTensor = tensor(facearray); % converts facearray from matrix format to tensor format

numdims = ndims(facearray);   % modes of tensors X_p; number of dimensions of input face array. numdims=4 for our case.

% set parameters 
opts = bcdLoR_als4; % name of the file containing algorithm
opts.MaxRecursivelevel = 1; 
opts.init = 'random';
opts.alsinit = 0;
opts.maxiters= 1500; % maximum interations allowed if criterion fit is not achieved
opts.printitn = 10; % how often printout iterations results

%%%%%% Main tensor decomposition algorithm here  %%%%%%%%%%%%
[Wthat,~,~] = bcdLoR_als4(facearrayTensor,LM,numdims,opts);

% Wthat is the important output for bcdLoR_als4, which is the tensor
% decomposition proper
%
% Wthat 4x1 cell array of factors (components): 
%1: image rows  x   total rank of all components
%2: image columns  x   total rank of all components
%3: color (3) X total rank of all components)
%4: numimages  x   num basis patterns   (mixing coefficients or weights)
% total rank of all components is the sum of ranks of all components. For
% example, if there are 20 components of rank=8, total rank is 160.

% convert BCD components to BCD tensor
V = cp2bcd(Wthat,LM,numdims);

% "component" stores tensor components in matrix. This is the important output
components = zeros(imagerows,imagecolumns,3,numcomponents);
weights = zeros(numcomponents,numfaces);
    
for k = 1:numcomponents
    components(:,:,:,k) = double(full(ktensor(V(1:3,k))));
    weights(k,:) = double(full(ktensor(V(4,k))));
end

%%%%%%%%%%%%%%% save results to file in directory 'components'  %%%%%%%%%%%%%%%%%%%
% name of the file identifies a. number of components   b. rank of
% components  c. number of faces that served as input to decomposition
resultsfilename = sprintf('component%drank%dfaceset%d',numcomponents,round(mean(L)),numfaces);

cd(componentsdirectory)
save(resultsfilename,'components','weights','sizefaces')
cd(mfilesdirectory)

s = fprintf('output in file %s in components directory\n',resultsfilename);

