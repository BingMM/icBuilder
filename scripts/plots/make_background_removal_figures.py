#%% Import 

import os
import numpy as np
import glob
import xarray as xr
import fuvpy as fuv
import matplotlib.pyplot as plt
from polplot import pp
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from pathlib import Path
from os.path import join as pjoin

#%% Paths

base = pjoin(Path(__file__).resolve().parents[1], 'example_data')

p_wic_nc = pjoin(base, 'wic')
p_s12_nc = pjoin(base, 's12')
p_s13_nc = pjoin(base, 's13')

p_wic_out = pjoin(base, 'figures', 'wic_br')
p_s12_out = pjoin(base, 'figures', 's12_br')
p_s13_out = pjoin(base, 'figures', 's13_br')

#%% Fetch orbits available in all nc files

# Fetch all orbits
o_wic = [int(o[-7:-3]) for o in sorted(glob.glob(p_wic_nc + '*.nc'))]
o_s12 = [int(o[-7:-3]) for o in sorted(glob.glob(p_s12_nc + '*.nc'))]
o_s13 = [int(o[-7:-3]) for o in sorted(glob.glob(p_s13_nc + '*.nc'))]

#%% Fun

def plot_br(inpath, outpath, orbits, sensor):
    for orbit in tqdm(orbits, total=len(orbits), desc=sensor):
        # Load nc orbit file for sensor
        so = xr.open_dataset(pjoin(inpath, sensor, '_or' + str(orbit).zfill(4) + '.nc')).copy()
        # Make folder
        foldername = pjoin(outpath, 'or_' + str(orbit).zfill(4))
        os.makedirs(foldername, exist_ok=True)
        print(foldername)
        # Loop over each date        
        for i in range(so.dims['date']):
            s = so.isel(date=i)
            figname = pjoin(foldername, '/t' + str(i).zfill(3) + '.png')
            make_plot(s, sensor, figname)

def make_plot(s, sensor, figname):
    
    fs0 = 20
    fs1 = 16
    fs2 = 12
    
    isensor = np.argmax(np.isin(['wic', 's13', 's12'], sensor))
    
    plt.ioff()
    fig = plt.figure(figsize=(15,9))
    gs = gridspec.GridSpec(nrows=2,ncols=4,hspace=0.3,wspace=0.01)
        
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')  # Turn off the axis for this subplot    
    ax.text(.45, .6, ['WIC', 'SI12', 'SI13'][isensor], fontsize=fs0, va='center', ha='center', transform=ax.transAxes)
    ax.text(.45, .5, 'Orbit ' + figname[-13:-9], fontsize=fs0, va='center', ha='center', transform=ax.transAxes)
    ax.text(.45, .4, np.datetime_as_string(s.date.values, 's'), fontsize=fs0, va='center', ha='center', transform=ax.transAxes)
        
    ## img ##
    vmin = np.array([0, 0, 0])[isensor]
    vmax = np.array([5000, 40, 20])[isensor]
    pax = pp(plt.subplot(gs[0,0]),minlat=50)
    #fuv.plotimg(s,'img',pax=pax,crange=(vmin,vmax),cmap='magma')
    fuv.plotimg(s,'img',pax=pax,crange=None,cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03])
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('Projected image [counts]', fontsize=fs1)
    pax.ax.set_title('BS',rotation='vertical', x=-0.055,y=0.48, va='center',ha='center')
    pax.write(50, 12, '12',va='bottom',ha='center',fontsize=fs2)
    pax.write(50, 18, '18',va='center',ha='right',fontsize=fs2)
    pax.write(50, 9, '50$^{\circ}$',va='center',ha='center',fontsize=fs2)

    # Dayglow
    pax = pp(plt.subplot(gs[0,1]),minlat=50)
    #fuv.plotimg(s,'dgmodel',pax=pax,crange=(vmin,vmax),cmap='magma')
    fuv.plotimg(s,'dgmodel',pax=pax,crange=None,cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('BS model [counts]', fontsize=fs1)
    pax.write(50, 12, '12',va='bottom',ha='center',fontsize=fs2)
    pax.write(50, 9, '50$^{\circ}$',va='center',ha='center',fontsize=fs2)

    # Corr
    #vmin = np.array([-1000, -15, -5])[isensor]
    #vmax = np.array([1000, 15, 5])[isensor]
    vmin = np.array([-5000, -40, -20])[isensor]
    vmax = np.array([5000, 40, 20])[isensor]
    pax = pp(plt.subplot(gs[0,2]),minlat=50)
    #fuv.plotimg(s,'dgimg',pax=pax,crange=(vmin,vmax),cmap='coolwarm')
    fuv.plotimg(s,'dgimg',pax=pax,crange=None,cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('BS corrected image [counts]', fontsize=fs1)
    pax.write(50, 12, '12',va='bottom',ha='center',fontsize=fs2)
    pax.write(50, 9, '50$^{\circ}$',va='center',ha='center',fontsize=fs2)

    #Weight
    pax = pp(plt.subplot(gs[0,3]),minlat=50)
    fuv.plotimg(s,'dgweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights', fontsize=fs1)
    pax.write(50,  6, '06',va='center',ha='left',fontsize=fs2)
    pax.write(50, 12, '12',va='bottom',ha='center',fontsize=fs2)
    pax.write(50, 9, '50$^{\circ}$',va='center',ha='center',fontsize=fs2)

    # Dayglow (SH)
    vmin = np.array([-1000, -15, -5])[isensor]
    vmax = np.array([1000, 15, 5])[isensor]
    pax = pp(plt.subplot(gs[1,1]),minlat=50)
    #fuv.plotimg(s,'shmodel',pax=pax,crange=(vmin,vmax),cmap='coolwarm')
    fuv.plotimg(s,'shmodel',pax=pax,crange=None,cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH model [counts]', fontsize=fs1)
    pax.ax.set_title('SH',rotation='vertical',x=-0.055,y=0.48,va='center',ha='center')
    pax.write(50, 12, '12',va='bottom',ha='center',fontsize=fs2)
    pax.write(50, 18, '18',va='center',ha='right',fontsize=fs2)
    pax.write(50, 9, '50$^{\circ}$',va='center',ha='center',fontsize=fs2)

    # Corr
    vmin = np.array([-5000, -40, -20])[isensor]
    vmax = np.array([5000, 40, 20])[isensor]
    pax = pp(plt.subplot(gs[1,2]),minlat=50)
    #fuv.plotimg(s,'shimg',pax=pax,crange=(vmin,vmax),cmap='coolwarm')
    fuv.plotimg(s,'shimg',pax=pax,crange=None,cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH corrected image [counts]', fontsize=fs1)
    pax.write(50, 12, '12',va='bottom',ha='center',fontsize=fs2)
    pax.write(50, 9, '50$^{\circ}$',va='center',ha='center',fontsize=fs2)

    #Weight
    pax = pp(plt.subplot(gs[1,3]),minlat=50)
    fuv.plotimg(s,'shweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights', fontsize=fs1)
    pax.write(50,  6, '06',va='center',ha='left',fontsize=fs2)
    pax.write(50, 12, '12',va='bottom',ha='center',fontsize=fs2)
    pax.write(50, 9, '50$^{\circ}$',va='center',ha='center',fontsize=fs2)

    plt.savefig(figname, bbox_inches='tight', dpi = 300)
    plt.close('all')
    plt.close()

#%%

plot_br(p_wic_nc, p_wic_out, o_wic, 'wic')
plot_br(p_s12_nc, p_s12_out, o_s12, 's12')
plot_br(p_s13_nc, p_s13_out, o_s13, 's13')





