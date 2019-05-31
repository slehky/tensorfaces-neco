function reconstructFaces(componentsfile,varargin)
% FUNCTION RECONSTRUCTFACES(COMPONENTSFILE,VARARGIN)
%
% Input parameter 1
% componentsfile: contains tensor components in directory "components"
%     If only one parameter, reconstructs original faces from decomposition
%     Example components file: components/component20rank8faceset128Example.mat
%     containing decomposition of faces into 20 components, rank=8, based on 128
%     faces as input. Componentsfile contains components, weights, and
%     sizefaces.
%
% Input parameter 2 (optional)
% faceInput: contains set of faces you want to reconstruct, different from
% original sample face set.
%     If second parameter exists, reconstructs different set of faces read
%     in from a file.
%     Example faces file containing 40 faces : testset40shuffled
%
%  Reconstruct original faces:
%  reconstructFaces('component20rank8faceset128Example')
%
%  Reconstruct different faces contained in face file testset40shuffled:
%  reconstructFaces('component20rank8faceset128Example','testset40shuffled')
%
% Sidney Lehky  2019

% Edit to specify your "homedirectory" in file "setupDirectories"
setupDirectories

if nargin == 0
     error('Enter name of components file')
end

if ~isempty(varargin)
    facesInput = varargin(1);
end


% Load tensor components
% The variable being loaded is "component".
% Weights for the original decomposition are loaded here also.
eval(['load ',char(componentsfile)]);

numpixels = numel(components(:,:,:,1)); % number of pixels in one face image

% vectorize component
components = reshape(components,numpixels,[]);

% reconstruct a different set of faces that has a different set of weights 
if ~isempty(varargin)
    % Load array of faces
    % The variable being loaded is "face"
    eval(['load ',char(facesInput)]);
    faceLAB = rgb2lab(face);  % face in LAB color face
    
    % vectorize faceLAB
    faceLAB = reshape(faceLAB,numpixels,[]);
    
    sizefaces = size(face);

    % Face decomposition: determine weights for components (tensorfaces) for
    % the facesInput set we wish to represent using the components.  
    % Components were generated in program "makeTensorFaces.m"
    weights = components\faceLAB;
end

% Face reconstruction: calculated from linear sum of weights * components
% Reconstruction for display of information contained in weights and components, does not actually occur in the brain.
% Typically reconstruct using test faces different from sample faces used to generate components. 
% Reconstruction error between original and reconstructed faces will depend on factors such as number of components.
reconstructedFaceLAB = components * weights;

reconstructedFaceLAB = reshape(reconstructedFaceLAB,sizefaces);  % in LAB color space
reconstructedFace = lab2rgb(reconstructedFaceLAB); % in RGB color space for display

% Show montage of reconstructed faces; don't have access of original faces
% unless we enter file as parameter 2.
figure(1)
montage(reconstructedFace)
set(gca,'FontSize',16)
title('Reconstructed faces')

if ~isempty(varargin) 
    % Show montage of original faces
    pause(1)
    figure(2)
    montage(face)
    set(gca,'FontSize',16)
    title('Original faces')
    
end