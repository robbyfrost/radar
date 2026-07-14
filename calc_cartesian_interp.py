# --------------------------------------------------
# Name: calc_cartesian_interp.py
# Author: Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 16 Jan 2026
# Purpose: Script to grid radar PPIs. Can do 3D Barnes
# or 2D Barnes with a linear interpolation in the 
# vertical dimension
# --------------------------------------------------
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
import sys
import glob
import os
import contextlib
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
sys.path.append('/home/robbyfrost/Analysis/TurbTor_Radar/')
from Analysis.TurbTor_Radar.radar_utils import calc_vort_radar
sys.path.append('/home/robbyfrost/Analysis/ADV_Cor/')
from ADVCor import *
# suppress pyart/pydda messages
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
with suppress_output():
    import pyart
    import pydda
# --------------------------------------------------
# parameters

# path to radar files
drad_ktlx = "/data/arrcwx/robbyfrost/20100510_mwr/ktlx_cfrad/"
dgrid_ktlx = "/data/arrcwx/robbyfrost/20100510_mwr/ktlx_grid/"
drad_oupr = "/data/arrcwx/robbyfrost/20100510_mwr/oupr_cfrad/"
dgrid_oupr = "/data/arrcwx/robbyfrost/20100510_mwr/oupr_grid/"
ddda = "/data/arrcwx/robbyfrost/20100510_mwr/dda/"
dsm = "/data/arrcwx/robbyfrost/20100510_mwr/oupr_sa/"

# vertical interpolation
self_interp = True
# make grid for wavelet
wavelet = False

# barnes parameters
dely, delx, delz = 200, 200, 100 # grid spacing in meters
Ly, Lx, Lz = 60000, 60000, 1100 # domain size in meters
ylim = (0,Ly)
xlim = (-Lx//2,Lx//2)
zlim = (0,Lz)
Ny, Nx, Nz = Ly//dely+1, Lx//delx+1, Lz//delz+1 # number of grid points
roi = 350.0
# --------------------------------------------------
# Find sweeps close in time

# OUPR
ffoupr = sorted(glob.glob(drad_oupr + "nexrad*"))
foupr = [f.split('/') for f in ffoupr]
foupr = [f[-1] for f in foupr]
# volume start time info
tstr_oupr = np.array([f[7:-11] for f in foupr])
toupr = np.array([datetime.strptime(tstr, "%Y%m%d_%H%M%S") for tstr in tstr_oupr], dtype="datetime64[ns]")
t0 = toupr[0]
t1 = toupr[-1]

# KTLX
ffktlx = sorted(glob.glob(drad_ktlx + "DEALIASED_KTLX*"))
fktlx = [f.split('/') for f in ffktlx]
fktlx = [f[-1] for f in fktlx]
# volume start time info
tstr_ktlx = np.array([f[14:-4] for f in fktlx])
tktlx = np.array([datetime.strptime(tstr, "%Y%m%d_%H%M%S") for tstr in tstr_ktlx], dtype="datetime64[ns]")
tktlx_idx = np.where((tktlx > (t0-np.timedelta64(5,'m'))) & (tktlx < (t1+np.timedelta64(5,'m'))))[0].astype(int)

# valid times for DDA
tktlx = tktlx[tktlx_idx]
tstr_ktlx = tstr_ktlx[tktlx_idx]
# --------------------------------------------------
# read in radar data

rktlx, roupr = [], []
for tstr in tstr_ktlx: # KTLX
    radar = pyart.io.read(f"{drad_ktlx}DEALIASED_KTLX{tstr}_V03")
    rktlx.append(radar)
for tstr in tstr_oupr: # OUPR
    radar = pyart.io.read(f"{drad_oupr}nexrad.{tstr}_OUPR.msg31")
    roupr.append(radar)
# --------------------------------------------------
# barnes interpolation

gktlx, goupr = [], []

# ktlx
# 3d barnes
if not self_interp and not wavelet:
    for jt in tqdm(range(len(rktlx))):
        grid = pyart.map.grid_from_radars(
            rktlx[jt],
            grid_shape=(Nz,Ny,Nx),
            weighting_function='barnes2',
            grid_limits=(
                zlim,
                ylim,
                xlim,
                ),
            grid_origin=(roupr[0].latitude['data'],
                         roupr[0].longitude['data']),
        )
        # output
        dout = f"{dgrid_ktlx}barnes_{delx}m_{tstr_ktlx[jt]}.nc"
        pyart.io.write_grid(dout, grid, format='NETCDF4')
        print(f"Barnes grid output to: {dout}")
        gktlx.append(grid)
# 2d barnes, linear interp in vertical
if self_interp and not wavelet:
    zgrid = np.arange(zlim[0], zlim[1]+1, delz)
    for jt in tqdm(range(len(rktlx))):
        radar = rktlx[jt]
        # remove survey mode sweeps
        ns = np.unique(radar.fixed_angle['data'], return_index=True)[1]
        for i in range(len(ns)-1):
            if (ns[i+1]-ns[i])//2:
                ns[i] = ns[i] + 1
        grid = pyart.map.grid_ppi_sweeps(
            radar,
            target_sweeps=ns,
            grid_size=(Ny,Nx),
            gridding_algo='map_gates_to_grid',
            weighting_function='barnes2',
            grid_limits=(
                (0,10),
                ylim,
                xlim,
            ),
            grid_origin=(roupr[0].latitude['data'][0],
                         roupr[0].longitude['data'][0]),
            roi_func="constant",
            constant_roi=roi,
        )
        # beam height
        proj = {
            'proj': grid.projection.attrs['proj'],
            'lon_0': roupr[jt].longitude['data'][0],
            'lat_0': roupr[jt].latitude['data'][0],
        }
        xk, yk = pyart.core.geographic_to_cartesian(
            radar.longitude['data'][0],
            radar.latitude['data'][0],
            proj
        )
        x = grid.x.values # OUPR X
        y = grid.y.values # OUPR Y
        X, Y = np.meshgrid(x, y, indexing='xy')
        r = np.sqrt((X - xk)**2 + (Y - yk)**2) # OUPR range

        Re = 6371000.0
        ke = 4.0 / 3.0
        elev = np.deg2rad(radar.fixed_angle['data'][ns])
        alt_corr = np.empty(grid.altitude_est.shape)
        for klev, theta in enumerate(elev):
            alt_corr[klev,:,:] = (
                np.sqrt(
                    r**2 + (ke * Re)**2 + 2.0 * r * ke * Re * np.sin(theta)
                ) 
                - ke * Re
            )
        grid.altitude_est.values[:] = alt_corr
        # interpolate z
        del radar
        # create new xarray dataset
        gridz = xr.Dataset(
            coords={
                'z': zgrid,
                'y': grid.y.values,
                'x': grid.x.values,
            }
        )
        # loop over fields
        for var in ['reflectivity', 'corrected_velocity', 'ROI']:
            data_interp = np.full((len(zgrid), Ny, Nx), np.nan)
            # loop over x,y
            for jy in range(Ny):
                for jx in range(Nx):
                    alt = grid.altitude_est[:,jy,jx].values
                    field = grid[var][:,jy,jx].values
                    data_interp[:,jy,jx] = np.interp(zgrid, 
                                                     alt,
                                                     field,
                                                     left=np.nan,
                                                     right=np.nan)
            # add field to dataset
            if var == "corrected_velocity":
                gridz['velocity'] = (('z', 'y', 'x'), data_interp)
            else:
                gridz[var] = (('z', 'y', 'x'), data_interp)
        # output
        dout = f"{dgrid_ktlx}barnes_{delx}m_{tstr_ktlx[jt]}_zlin_interp.nc"
        if os.path.exists(dout):
            os.remove(dout)
        gridz.to_netcdf(dout)
        print(f"Barnes grid output to: {dout}")
        gktlx.append(grid)

# oupr
# 3d barnes
if not self_interp and not wavelet:
    for jt in tqdm(range(len(roupr))):
        grid = pyart.map.grid_from_radars(
            roupr[jt],
            grid_shape=(Nz,Ny,Nx),
            gridding_algo='map_gates_to_grid',
            weighting_function='barnes2',
            grid_limits=(
                zlim,
                ylim,
                xlim,
                ),
        )
        dout = f"{dgrid_oupr}barnes_{delx}m_{tstr_oupr[jt]}.nc"
        pyart.io.write_grid(dout, grid, format='NETCDF4')
        print(f"Barnes grid output to: {dout}")
        goupr.append(grid)
# 2d barnes, linear interp in vertical
if self_interp and not wavelet:
    zgrid = np.arange(zlim[0], zlim[1]+1, delz)
    for jt in tqdm(range(len(roupr))):
        radar = roupr[jt]
        grid = pyart.map.grid_ppi_sweeps(
            radar,
            grid_size=(Ny,Nx),
            gridding_algo='map_gates_to_grid',
            weighting_function='barnes2',
            grid_limits=(
                (0,10),
                ylim,
                xlim,
            ),
            roi_func="constant",
            constant_roi=roi,
        )
        # beam height
        x = grid.x.values # OUPR X
        y = grid.y.values # OUPR Y
        X, Y = np.meshgrid(x, y, indexing='xy')
        r = np.sqrt((X)**2 + (Y)**2) # OUPR range

        Re = 6371000.0
        ke = 4.0 / 3.0
        elev = np.deg2rad(radar.fixed_angle['data'])
        alt_corr = np.empty(grid.altitude_est.shape)
        for klev, theta in enumerate(elev):
            alt_corr[klev,:,:] = (
                np.sqrt(
                    r**2 + (ke * Re)**2 + 2.0 * r * ke * Re * np.sin(theta)
                ) 
                - ke * Re
            )
        grid.altitude_est.values[:] = alt_corr
        # interpolate z
        del radar
        # create new xarray dataset
        gridz = xr.Dataset(
            coords={
                'z': zgrid,
                'y': grid.y.values,
                'x': grid.x.values,
            }
        )
        # loop over fields
        for var in ['reflectivity', 'velocity', 'ROI']:
            data_interp = np.full((len(zgrid), Ny, Nx), np.nan)
            # loop over x,y
            for jy in range(Ny):
                for jx in range(Nx):
                    alt = grid.altitude_est[:,jy,jx].values
                    field = grid[var][:,jy,jx].values
                    # remove NaNs
                    # mask = ~np.isnan(field) & ~np.isnan(alt)
                    # if np.any(mask):
                    data_interp[:,jy,jx] = np.interp(zgrid, 
                                                     alt,#[mask], 
                                                     field,#[mask],
                                                     left=np.nan,
                                                     right=np.nan)
            # add field to dataset
            gridz[var] = (('z', 'y', 'x'), data_interp)
        # output
        dout = f"{dgrid_oupr}barnes_{delx}m_{tstr_oupr[jt]}_zlin_interp.nc"
        if os.path.exists(dout):
            os.remove(dout)
        gridz.to_netcdf(dout)
        print(f"Barnes grid output to: {dout}")
        goupr.append(grid)

# OUPR barnes for wavelet analysis
if wavelet:
    for jt in tqdm(range(len(roupr))):
        radar = roupr[jt]
        radar = calc_vort_radar(radar, 'velocity', 'vorticity', 1, 1)

        grid = pyart.map.grid_ppi_sweeps(
            radar,
            grid_size=(Ny,Nx),
            gridding_algo='map_gates_to_grid',
            weighting_function='barnes2',
            grid_limits=(
                (0,10),
                ylim,
                xlim,
            ),
            roi_func="constant",
            constant_roi=roi,
        )
        
        # output
        dout = f"{dgrid_oupr}barnes_swp_{delx}m_{tstr_oupr[jt]}.nc"
        if os.path.exists(dout):
            os.remove(dout)
        grid.to_netcdf(dout)
        print(f"Barnes grid output to: {dout}")
        goupr.append(grid)