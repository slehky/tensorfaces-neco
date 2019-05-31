%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2014
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

disp('__________________________________________________________________________') 
disp('                                                                          ') 
disp('  ______                 |    TTeMPS: A TT/MPS tensor toolbox for MATLAB  ')
disp('   |  |   |\/||_)(_`     |                                                ') 
disp('   |  | E |  ||  __)     |    Michael Steinlechner                        ')
disp('                         |    Ecole Polytechnique Federale de Lausanne    ')
disp('                                                                          ')
disp('__________________________________________________________________________')
disp('   ')
disp('         This toolbox is designed to simplify algorithmic development in the ')
disp('         TT/MPS format, making use of the object oriented programming ')
disp('         programming techniques introduced in current MATLAB versions. ')
disp('   ')
disp('WARNING: TTeMPS is experimental and not a finished product. ')
disp('         Many routines do not have sanity checks for the inputs. Use with care. ')
disp('         For questions and contact: michael.steinlechner@epfl.ch                ')
disp('                                                                               ')
disp('         For a much more complete toolbox for this tensor format, we refer to the')
disp('         TT Toolbox by Oseledets et al., https://github.com/oseledets/TT-Toolbox.')
disp(' ')
disp('         In this toolbox, you will also find conversion routines betweens TTeMPS ')
disp('         and the TT Toolbox. If these are needed, the TT toolbox has to be loaded')
disp('         into the current path, too.')
disp(' ')
disp('TTeMPS is licensed under a BSD 2-clause license, see LICENSE.txt')
disp('   ')
                                   
addpath( cd )
addpath( [cd, filesep, 'algorithms'] )
addpath( [cd, filesep, 'operators'] )
addpath( [cd, filesep, 'examples'] )

disp('Finished. Try out the example code example.m')








