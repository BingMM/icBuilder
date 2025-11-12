#%% Import 

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from polplot import pp
from tqdm import tqdm

from icreader import ConductanceImage

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
#base = '/Home/siv32/mih008/repos/icBuilder/example_data/'
#base = '/disk/IMAGE_FUV/fuv/'

p_in = base + 'conductance/'
p_out = base + 'figures/conductance/'

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Func

def get_c_scales(cI):
    c_scales = {'wicm': (0, np.round(np.nanmax(cI.wic_avg)+1)),
                'wics': (0, np.round(np.nanmax(cI.wic_std)+1)),
                's12m': (0, np.round(np.nanmax(cI.s12_avg)+1)),
                's12s': (0, np.round(np.nanmax(cI.s12_std)+1)),
                's13m': (0, np.round(np.nanmax(cI.s13_avg)+1)),
                's13s': (0, np.round(np.nanmax(cI.s13_std)+1)),
                'E0':   (0,  25),
                'dE0':  (0, np.round(np.nanmax(cI.dE0)+1)),
                'Fe':   (0, np.round(np.nanmax(cI.Fe)+1)),
                'dFe':  (0, np.round(np.nanmax(cI.dFe)+1)),
                'R':    (0, 150),
                'dR':   (0, np.round(5*np.median(cI.dR[~np.isnan(cI.dR)])+1)),
                'H':    (0, np.round(np.nanmax(cI.H)+1)),
                'dH':   (0, np.round(np.nanmax(cI.dH)+1)),
                'P':    (0, np.round(np.nanmax(cI.P)+1)),
                'dP':   (0, np.round(np.nanmax(cI.dP)+1))
                }
    return c_scales
    
def plot(cI, i, c_scales, lat, lt):    
        
    fig, axs = plt.subplots(3, 6, figsize=(30, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axs.flatten()[:-2]
    axs[2, 4].set_axis_off()
    axs[2, 5].set_axis_off()
    
    var = [cI.wic_avg[i], cI.wic_std[i], cI.R[i],  cI.dR[i],  cI.H[i], cI.dH[i],
           cI.s13_avg[i], cI.s13_std[i], cI.E0[i], cI.dE0[i], cI.P[i], cI.dP[i],
           cI.s12_avg[i], cI.s12_std[i], cI.Fe[i], cI.dFe[i]]
    
    cs = [c_scales['wicm'], c_scales['wics'], c_scales['R'],  c_scales['dR'],  c_scales['H'], c_scales['dH'],
          c_scales['s13m'], c_scales['s13s'], c_scales['E0'], c_scales['dE0'], c_scales['P'], c_scales['dP'],
          c_scales['s12m'], c_scales['s12s'], c_scales['Fe'], c_scales['dFe']]
    
    tit = ['avg WIC counts', 'std WIC counts', 'WIC*/S13* (R)', 'R std', 'Hall', 'Hall std',
            'avg S13 counts', 'std S13 counts', 'E0', 'E0 std', 'Pedersen', 'Pedersen std',
            'avg S12 counts', 'std S12 counts', 'Fe', 'Fe std']
    
    for j, (ax, var_, cs_, tit_) in enumerate(zip(axes, var, cs, tit)):
        pax = pp(ax)
        if j == 12:
            pax.writeLTlabels(fontsize=16)
            ax.text(.85, .1, '50$^{\circ}$', ha='center', va='center', fontsize=16, transform=ax.transAxes)
        pax.plotimg(lat, lt, var_, crange=cs_)
        ax.set_title(tit_, fontsize=18)
        ax.text(.85, .85, str(int(cs_[-1])), ha='left', va='center', fontsize=16, transform=ax.transAxes)
    
    axs[0,2].text(1.1, 1.2, cI.time[i], ha='center', va='center', fontsize=20, transform=axs[0,2].transAxes)
    
#%% Plot

plt.ioff()
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    
    cI = ConductanceImage(filename)
    
    c_scales = get_c_scales(cI)
    
    lat = cI.grid.lat
    lt = (cI.grid.lon/15)%24
    
    p_out_o = p_out + f'or_{str(orbit).zfill(4)}/'
    os.makedirs(p_out_o, exist_ok=True)
    
    for i in range(cI.shape[0]):
        try:
            plot(cI, i, c_scales, lat, lt)
            plt.savefig(p_out_o + f'{str(i).zfill(3)}.png', bbox_inches='tight')
        except:
            print(f'Plot failed: orbit {orbit}, index {i}')
        plt.close('all')
