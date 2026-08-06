#%% Import 

import os
from os.path import join as pjoin
import numpy as np
import glob
import matplotlib.pyplot as plt
from polplot import pp
from tqdm import tqdm
from pathlib import Path
from icreader import ConductanceImage
import argparse

#%% Argument parsing

parser = argparse.ArgumentParser(description="Base path.")
parser.add_argument('--base', type=str,
                    default=str(pjoin(Path(__file__).resolve().parents[1], 'example_data')),
                    help='Base data directory')
args = parser.parse_args()

#%% Paths

base = args.base

p_in  = pjoin(base, 'conductance')
p_out = pjoin(base, 'figures', 'conductance')

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(pjoin(p_in, '*.nc')))]

#%% Func

def get_max(x, quan=.999):
    return np.round(np.nanquantile(x, quan)+1)

def get_c_scales(cI):
    c_scales = {'wicm': (0, get_max(cI.wic_avg)),
                'wics': (0, get_max(cI.wic_std)),
                's12m': (0, get_max(cI.s12_avg)),
                's12s': (0, get_max(cI.s12_std)),
                's13m': (0, get_max(cI.s13_avg)),
                's13s': (0, get_max(cI.s13_std)),
                'E0':   (0,  10),
                'dE0':  (0, get_max(cI.dE0)),
                'Fe':   (0, get_max(cI.Fe)),
                'dFe':  (0, get_max(cI.dFe)),
                'R':    (0, 150),
                'dR':   (0, np.round(5*np.median(cI.dR[~np.isnan(cI.dR)])+1)),
                'H':    (0, get_max(cI.H)),
                'dH':   (0, get_max(cI.dH)),
                'P':    (0, get_max(cI.P)),
                'dP':   (0, get_max(cI.dP)),
                'H/P':  (0, get_max(.45*cI.E0**.85))}
    return c_scales
    
def plot(cI, i, c_scales, lat, lt):    
        
    fig, axs = plt.subplots(3, 6, figsize=(30, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axs.flatten()[:-1]
    axs[2, 5].set_axis_off()
    
    var = [cI.wic_avg[i], cI.wic_std[i], cI.R[i],  cI.dR[i],  cI.H[i], cI.dH[i],
           cI.s13_avg[i], cI.s13_std[i], cI.E0[i], cI.dE0[i], cI.P[i], cI.dP[i],
           cI.s12_avg[i], cI.s12_std[i], cI.Fe[i], cI.dFe[i], .45*cI.E0[i]**.85]
    
    cs = [c_scales['wicm'], c_scales['wics'], c_scales['R'],  c_scales['dR'],  c_scales['H'], c_scales['dH'],
          c_scales['s13m'], c_scales['s13s'], c_scales['E0'], c_scales['dE0'], c_scales['P'], c_scales['dP'],
          c_scales['s12m'], c_scales['s12s'], c_scales['Fe'], c_scales['dFe'], c_scales['H/P']]
    
    tit = ['avg WIC counts', 'std WIC counts', 'WIC*/S13* (R)', 'R std', 'Hall', 'Hall std',
            'avg S13 counts', 'std S13 counts', 'E0', 'E0 std', 'Pedersen', 'Pedersen std',
            'avg S12 counts', 'std S12 counts', 'Fe', 'Fe std', 'H/P ratio']
    
    for j, (ax, var_, cs_, tit_) in enumerate(zip(axes, var, cs, tit)):
        pax = pp(ax)
        if j == 12:
            pax.writeLTlabels(fontsize=16)
            ax.text(.85, .1, '50$^{\circ}$', ha='center', va='center', fontsize=16, transform=ax.transAxes)
        pax.plotimg(lat, lt, var_, crange=cs_)
        ax.set_title(tit_, fontsize=18)
        ax.text(.85, .85, str(int(cs_[-1])), ha='left', va='center', fontsize=16, transform=ax.transAxes)
    
    if hasattr(cI, 'Kp'):        
        stit = 'Kp: ' + str(cI.Kp[i]) + ' - ' + str(cI.time[i])
    else:
        stit = cI.time[i]
    axs[0,2].text(1.1, 1.2, stit, ha='center', va='center', fontsize=20, transform=axs[0,2].transAxes)
    #axs[0,2].text(1.1, 1.2, cI.time[i], ha='center', va='center', fontsize=20, transform=axs[0,2].transAxes)
    
#%% Plot

plt.ioff()
for orbit in tqdm(o, total=len(o)):
    filename = pjoin(p_in, f'or_{str(orbit).zfill(4)}.nc')
    print(filename)

    cI = ConductanceImage(filename)
    
    c_scales = get_c_scales(cI)
    
    lat = cI.grid.lat
    lt = (cI.grid.lon/15)%24
    
    p_out_o = pjoin(p_out, f'or_{str(orbit).zfill(4)}')
    os.makedirs(p_out_o, exist_ok=True)
    
    for i in range(cI.shape[0]):
        try:
            plot(cI, i, c_scales, lat, lt)
            plt.savefig(pjoin(p_out_o, f'{str(i).zfill(3)}.png'), bbox_inches='tight')
        except:
            print(f'Plot failed: orbit {orbit}, index {i}')
        plt.close('all')
