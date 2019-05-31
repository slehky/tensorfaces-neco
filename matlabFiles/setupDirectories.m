
% (Assumes Linux or Mac directory names, easily modified for Windows)

% ****** Modify your own own homedirectory to point to tensorfaces-neuralcomputation directory that was downloaded ******* 
homedirectory = '/Users/sidney/FaceComponents/tensorfaces-neco/';
% *******************************************************************************

mfilesdirectory = strcat(homedirectory,'matlabfiles');
facesdirectory = strcat(homedirectory,'faces/');
componentsdirectory = strcat(homedirectory,'components/');

cd(homedirectory)
if ~exist(componentsdirectory,'dir')
    mkdir('components')
end

toolbox1 = strcat(homedirectory,'matlabToolboxes/tensor_toolbox');
toolbox2 = strcat(homedirectory,'matlabToolboxes/TensorBox');

addpath(mfilesdirectory,facesdirectory,componentsdirectory,toolbox1,toolbox2)
