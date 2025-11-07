#%% Import

import os
from icreader import ConductanceImage
from icbuilder import SplineImage

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit_file = 'or_0086.nc'
conductance_file = os.path.join(base, 'conductance', orbit_file)
spline_file = os.path.join(base, 'spline', orbit_file)

#%% Load condutance image

cI = ConductanceImage(conductance_file)

#%% Create spline image

sI = SplineImage(cI)

#%% Save spline image to nc

sI.to_nc(spline_file)

