function displayTensorComponents(componentsfile)
% FUNCTION DISPLAYTENSORCOMPONENTS(COMPONENTSFILE)
% This produces a montage of all tensor component images
%
% This program rescales the three color components of the LAB color space solely for
% display purposes, to fit the color space. The raw color components may be
% too small or too large (beyond the allowable ranges allowed by the color space) for visualization.
%
% Example components file: components/component20rank8faceset128Example.mat
% containing decomposition of faces into 20 components, rank=8, based on 128
% faces as input.
%
% displayTensorComponents('component20rank8faceset128Example')
%
% Sidney Lehky 2019


  % Edit to specify your "homedirectory" in file "setupDirectories"
  setupDirectories
  
  % load components file
  cd(componentsdirectory)
  eval(['load ',char(componentsfile)]);
  
  % components is contained in the input file
  imagerows = size(components,1);
  imagecolumns = size(components,2);
  numcomponents = size(components,4);
  
  componentimage = zeros(imagerows,imagecolumns,3,numcomponents);
  for k = 1:numcomponents
     comp = reshape(components(:,:,:,k),[],1);
    
     % get the sign of the max absolute value of component 
     [~,maxabsval] = max(abs(comp));
     signcomp = sign(comp(maxabsval));
    
     % reverse sign of component if has negative values so that component
     % can be displayed in RGB color format
     if signcomp >= 0
        componentimage(:,:,:,k) = components(:,:,:,k);
     else
        componentimage(:,:,:,k) = -components(:,:,:,k);
     end
  end
      
  componentimage = lab2rgbnorm2(componentimage);
  componentimage = reshape(componentimage,imagerows,imagecolumns,3,[]);
  

  montage(componentimage);
  
end



function tensorComponentsRGB = lab2rgbnorm2(tensorComponents)
% FUNCTION IMBCDRGB = LAB2RGBNORM2(IMBCD)
% Normalizes luminance and color for each tensor component in order to expand color range for display. 
% Then converts LAB to RGB color space. This is only for displaying tensor
% component images and has no effect on calculations of tensor processing.
% Components were originally generated in LAB color space beause it is more
% physiological than RGB space.

   numrows = size(tensorComponents,1); % face image size in pixels
   numcolumns = size(tensorComponents,2); % face image size in pixels
   numcomponents = size(tensorComponents,4); % number of tensor components

   tensorComponentsNorm = zeros(numrows,numcolumns,3); % normalized LAB tensor components
   tensorComponentsRGB = zeros(numrows,numcolumns,3,numcomponents); % normalized RGB tensor components for display

   % For each tensor component
   for i=1:numcomponents
       % Process luminance and color channels of LAB colorspace separately
       chan1 = tensorComponents(:,:,1,i); % luminance channel 1
       chan23 = tensorComponents(:,:,2:3,i);  % color channels 2 and 3
    
       % normalize luminance channel with minimum of image set to zero for display;
       % luminance channel can't have negative values
       chan1 = chan1 - min(chan1(:));
 
       % flips luminance range to correct for "negative" luminances permitted
       % by algorithm 
       m = max(chan1(:));
       n = min(chan1(:));
       if abs(n) > abs(m)
           chan1 = -chan1;
       end
    
       % LAB luminance channel extends over the range 0 to 100
       k = (100-0)/(m-n);
       j = 0-k*n;
    
       % do linear transform of luminance to normalize it
       chan1 = k .* chan1 + j;
    
       
       % Normalize the two LAB color channels
       % flips color range to correct for "negative" colors permitted
       % by algorithm 
       m = max(chan23(:));
       n = min(chan23(:));
       if abs(n) > abs(m)
           chan23 = -chan23;
       end
    
       % LAB color channels extend over the range -100 to 100
       k = (100 - (-100))/(m-n);
       j = -100-k*n;
       chan23 = k.*chan23 + j;

       % combine luminance and color channels into one array
       tensorComponentsNorm(:,:,1) = chan1;  % luminance
       tensorComponentsNorm(:,:,2:3) = chan23;  % color

       % convert LAB to RGB color space
       tensorComponentsRGB(:,:,:,i) = lab2rgb(tensorComponentsNorm,'ColorSpace','adobe-rgb-1998');
   end

end


