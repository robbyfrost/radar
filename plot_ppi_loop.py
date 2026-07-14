# --------------------------------------------------
# Name: plot_ppi_loop.py
# Author: Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 30 May 2025
# Purpose: Loop over radar files and plot a bunch of
# PPIs at once
# --------------------------------------------------
# Set up
# --------------------------------------------------
import pyart
import numpy as np
import xarray as xr
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
import sys
import glob
sys.path.append('/home/robbyfrost/Analysis/TurbTor_Radar/')
from Analysis.TurbTor_Radar.radar_utils import *

# plotting set up
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=15)
rc('figure', facecolor='white')
# --------------------------------------------------
# read in data
# --------------------------------------------------
# path to output figures
dfig = "/home/robbyfrost/Figures/raxpol/20250605/"
# read in radar object
drad = "/data/arrcwx/wrt_jtrf/morton/grf/bf2/"
# loop over files
for file in sorted(glob.glob(f"{drad}DEALIASED*")):
    # read radar file
    radar = pyart.io.read(file)
    # time information
    yr, mo, da, hr, mi, se, tstring, tdt = get_time_pyart(radar.time)

    # extract variables
    swp = 0
    el = radar.fixed_angle['data'][0]
    x, y, z = radar.get_gate_x_y_z(swp)
    radswp = radar.extract_sweeps([swp])
    snrh = radswp.fields['SNRH']['data']
    ref = radswp.fields['DBZ_ADJ']['data']
    ref = np.where(snrh > 0, ref, np.nan)
    ref_plot = np.ma.masked_less(ref, 0)
    vr = radswp.fields['VELDEL']['data']
    vr = np.where(snrh > 0, vr, np.nan)
    zdr = radswp.fields['ZDR']['data']
    zdr = np.where(snrh > 0, zdr, np.nan)
    rhohv = radswp.fields['RHOHV']['data']
    rhohv = np.where(snrh > 0, rhohv, np.nan)

    # # read in vortex positions
    # da = "/data/arrcwx/robbyfrost/raxpol/20250518/analysis/"
    # dvp = "combined_vortex_positions.txt"
    # df = pd.read_csv(f"{da}{dvp}")
    # vtime = df['# Time [YYYYMMDD_HHMMSS UTC]'].values
    # # find vortex position at closest time to radar
    # vtime_dt = np.array([datetime.strptime(vtime[jt], '%Y%m%d_%H%M%S') for jt in range(vtime.size)])
    # close_vort_time = min(vtime_dt, key=lambda x: abs(x - tdt))
    # close_vort_idx = np.where(vtime_dt == close_vort_time)[0][0]
    # # vortex lat/lon
    # vlat = float(df[' Vortex latitude'].values[close_vort_idx])
    # vlon = float(df[' Vortex longitude'].values[close_vort_idx])

    # --------------------------------------------------
    # plot PPI of Z, RHOHV, VR, VORTZ
    # --------------------------------------------------
    print(f"Plotting {tstring}")
        
    xlims = [-7.5,12.5]
    ylims = [0,20]

    fig, axs = plt.subplots(figsize=(21,18.5), 
                            ncols=2,
                            nrows=2,
                            sharey=True,
                            sharex=True,
                            constrained_layout=True,
                            dpi=200)

    fig.suptitle(f"RaXPol {tdt} UTC (El={el:.1f}$^{{\\circ}}$)", 
                fontsize=25, 
                fontweight='bold')

    # reflectivity
    ax = axs[0,0]
    vmin, vmax = 0, 60
    ax.set_title("(a) Equivalent Reflectivity Factor", loc='left')
    pcm = ax.pcolormesh(
        x/1e3,
        y/1e3,
        ref_plot,
        vmin=vmin,
        vmax=vmax,
        cmap='Carbone42'
        )
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, label="$Z$ [dBZ]")
    cbar.set_ticks(np.arange(vmin, vmax+0.1, 10))
    ax.set_ylabel("Meridional Distance [km]")
    # radial velocity
    ax = axs[0,1]
    vmin, vmax = -50, 50
    ax.set_title("(b) Radial Velocity", loc='left')
    pcm = ax.pcolormesh(
        x/1e3,
        y/1e3,
        vr,
        vmin=vmin,
        vmax=vmax,
        cmap='Carbone42'
        )
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, label="$V_r$ [m s$^{-1}$]")
    cbar.set_ticks(np.arange(vmin, vmax+0.1, 10))
    # differential reflectivity
    ax = axs[1,0]
    vmin, vmax = -3, 8
    ax.set_title("(c) Differential Reflectivity", loc='left')
    pcm = ax.pcolormesh(
        x/1e3,
        y/1e3,
        zdr-4,
        vmin=vmin,
        vmax=vmax,
        cmap='Carbone42'
        )
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, label="$Z_{DR}$ [dB]")
    cbar.set_ticks(np.arange(vmin, vmax+0.001, 2))
    pcm.set_rasterized(True)
    ax.set_xlabel("Zonal Distance [km]")
    ax.set_ylabel("Meridional Distance [km]")
    # correlation coefficient
    ax = axs[1,1]
    vmin, vmax = 0.5, 1
    ax.set_title("(d) Cross-Correlation Ratio", loc='left')
    pcm = ax.pcolormesh(
        x/1e3,
        y/1e3,
        rhohv,
        vmin=vmin,
        vmax=vmax,
        cmap='Carbone42'
        )
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, label="$\\rho_{HV}$")
    cbar.set_ticks(np.arange(vmin, vmax + 0.001, 0.1))
    ax.set_xlabel("Zonal Distance [km]")

    # clean up plot
    for ax in axs.flatten():
        ax.set_aspect('equal')
        ax.set_xlim(xlims)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylim(ylims)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(alpha=0.6)

    # finish up plot
    dout = f"{dfig}{tstring}_z_rhohv_vel_zeta_{el:.1f}el.png"
    plt.savefig(dout, bbox_inches='tight')
    print(f"Saved to: {dout}\n")
    plt.close(fig)

    # # --------------------------------------------------
    # # plot PPI of Z, VR
    # # --------------------------------------------------
    # el = float(radar.elevation['data'][-1])
    # if (el < 5) and (el > 1):
    #     print(f"Plotting {tstring}")
    #     display = pyart.graph.RadarDisplay(radar.extract_sweeps([0]))
    #     xlims = [-7.5,7.5]
    #     ylims = [0,15]

    #     fig, axs = plt.subplots(figsize=(16,9.25), ncols=2, nrows=1, constrained_layout=True)
    #     fig.suptitle(f"{radar.metadata['instrument_name'][:-4]} {tdt} UTC (El={round(float(radar.elevation['data'][-1]),1)}$^{{\\circ}}$)", fontsize=25, fontweight='bold')

    #     # reflectivity
    #     ax = axs[0]
    #     vmin, vmax = -10, 70
    #     display.plot_ppi('DBZ', ax=ax, 
    #                     vmin=vmin, vmax=vmax,
    #                     cmap="pyart_Carbone42",
    #                     colorbar_flag=False,
    #                     title_use_sweep_time=False,
    #                     axislabels_flag=False)
    #     ax.set_title("Equivalent Reflectivity Factor")
    #     cbar0 = fig.colorbar(display.plots[0], ax=ax, orientation='horizontal', pad=0.015)
    #     cbar0.set_label("$Z$ [dBZ]")
    #     cbar0.set_ticks(np.arange(vmin, vmax + 0.001, 10))
    #     # radial velocity
    #     ax = axs[1]
    #     vmin, vmax = -50, 50
    #     display.plot_ppi('VEL_DEALIAS', ax=ax, 
    #                     vmin=vmin, vmax=vmax,
    #                     cmap="pyart_Carbone42",
    #                     colorbar_flag=False,
    #                     title_use_sweep_time=False,
    #                     axislabels_flag=False)
    #     ax.set_title("Dealiased Radial Velocity")
    #     cbar2 = fig.colorbar(display.plots[1], ax=ax, orientation='horizontal', pad=0.015)
    #     cbar2.set_label("$V_r$ [m s$^{-1}$]")
    #     cbar2.set_ticks(np.arange(vmin, vmax + 0.001, 10))
    #     # clean up plot
    #     for iax in axs.flatten():
    #         iax.set_aspect('equal')
    #         iax.set_xlim(xlims)
    #         iax.xaxis.set_major_locator(MultipleLocator(2))
    #         iax.xaxis.set_minor_locator(MultipleLocator(1))
    #         iax.set_xlabel("Zonal Distance [km]")
    #         iax.set_ylim(ylims)
    #         iax.yaxis.set_major_locator(MultipleLocator(2))
    #         iax.yaxis.set_minor_locator(MultipleLocator(1))
    #         iax.grid(alpha=0.6)
    #         iax.set_ylabel("Meridional Distance [km]")

    #     # finish up plot
    #     dout = f"{dfig}{tstring}_z_vel_{round(el,1)}EL.png"
    #     plt.savefig(dout, dpi=150, bbox_inches='tight')
    #     print(f"Saved to: {dout}\n")
    #     plt.close(fig)
    # else:
    #     print(f"Skipping {tstring}\n")